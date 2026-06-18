# 學習記錄草稿（對外貼文）

> 連結：
> - GitHub repo：<https://github.com/Kaminoikari/physical-ai-agent>
> - fine-tune 的 VLA 模型：<https://huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft>

---

## 長版

### Physical AI 學習記錄：把編排層與 policy 解耦，並驗證它

這是一個關於 embodied AI 編排層的 side project。出發點是一個問題：VLA（Vision-Language-Action）模型已經能直接吃語言指令，為什麼還需要在它上面疊一層 LLM？

答案在於操作粒度。VLA 的輸出是單一原子動作（end-effector 的位移、夾爪開合），它把「看到的畫面 + 一句指令」映射到下一個控制訊號。但真實任務很少是單一原子動作：「把桌上的東西收進抽屜」需要先開抽屜、逐一抓取、放入、關上；順序錯了就失敗。決定這個順序、判斷某個子目標是否超出手臂能力、在指令有歧義時停下來反問，這些是規劃與決策，不是逐幀的動作映射。把它們塞進 policy 會讓 policy 同時背負兩種差異很大的職責；拆出來放到上層，policy 只需專注把單一技能做好。這個專案把這層（L3）實作出來，並驗證它與底層 policy（L1）解耦——也就是底層換模型、換環境時，上層邏輯不必跟著改。

**架構。** 三層，各自職責清楚：

- **L3（編排，Claude）** 接收自然語言任務，拆成有序的技能呼叫，維護「已完成哪些子目標」的狀態，處理重試與中止。它不碰控制訊號，只決定「下一步該呼叫哪個技能、帶什麼參數」。
- **L2（技能介面）** 是 L3 與 L1 之間的合約，只暴露兩個動詞：`execute`（去做一件事）與 `query`（問環境現在的狀態）。L3 永遠只透過這兩個介面跟下層互動，看不到底層是 mock 還是真模擬。
- **L1（policy + 模擬）** 是 SmolVLA 在 LIBERO 模擬裡跑單一抓放，把 L2 的 `execute` 落實成實際的 rollout。

這個分層的價值在替換成本：同一套 L3 編排骨架在 mock 實作與真模擬實作之間共用，從「假技能」切到「真模擬」只需替換 L2 的實作，L3 的迴圈、重試、中止邏輯一行不改。這正是「解耦」要驗證的事——介面穩定，兩邊各自演進。

**行為驗證。** 我刻意把驗證重點放在邊界情況，而不是挑一個漂亮的成功案例展示。因為 demo 一次成功不能證明系統可靠，真正要確認的是它在「不該做的時候不做」：

- **多步拆解與排序**——一句複合指令能被拆成正確順序的子任務。
- **超出能力時拒絕**——要求做手臂技能集裡沒有的動作時，L3 應該拒絕而不是硬湊一個會失敗的呼叫。
- **指令歧義時反問**——指涉不明（例如場上有多個同類物件）時停下來澄清，而非任意猜一個。
- **長程任務失敗後中止並回報**——官方 checkpoint 在該 suite 的長程任務上實測成功率僅約 5%，所以系統必須能辨識「這條路走不通」、乾淨地中止並說明原因，而不是無限重試。

這些重試與中止路徑都由單元測試固定下來，外部依賴（Claude、模擬器）全部 mock，所以判定不依賴單次 rollout 的運氣，每次跑結果一致。

**真機資料微調。** 編排層之上已經驗證，但「真實機器人資料 → policy」這一段在純模擬裡是缺的。為了把這段補上，我用社群公開的 SO-101 遙操作資料集（`lerobot/svla_so101_pickplace`，50 段示範、約 1.2 萬幀、雙相機視角），在 Kaggle 上 fine-tune SmolVLA（450M 參數的 VLA）。

過程中遇到的不是「能不能訓練」，而是資料與模型介面對齊的細節：資料集是兩個相機視角（up / side），但 `smolvla_base` 預期三路相機（camera1/2/3），直接訓練會在特徵驗證時報 mismatch。解法是用 `rename_map` 把資料集的視角映射到模型預期的子集（資料視角 ⊆ 模型視角即可通過驗證），而不是去動模型結構。硬體上 Kaggle 的 T4（16GB）吃不下官方建議的 batch size，降到 8 才不 OOM；2000 步在 T4 約 78 分鐘，loss 從 0.410 收到 0.141。產出的 checkpoint 已上傳並附完整 model card（[Kaminoikari/smolvla-so101-pickplace-ft](https://huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft)）。這一步確認的是整條資料到模型的流程能跑通，全程公開資料、零實體硬體。

**執行管線。** 最初的實作每個 LIBERO 任務都透過 subprocess 重新載入一次 policy——這在多任務評估時很浪費，因為 450M 模型的載入成本被重複付了 N 次。我把它重構成一個 `RolloutEngine` 介面，底下兩個實作：`SubprocessRolloutEngine`（舊路徑，每任務重載，留作 parity 基準）與 `InProcessRolloutEngine`（policy 與 env 載一次後常駐複用，直接呼叫 `eval_policy()`）。policy 與環境的建構抽成一個由 `(policy_type, policy_path)` 參數化的 builder，所以從 SmolVLA 換到 GR00T 只需換 path，不必改引擎邏輯。

重構正確性的關鍵是 parity：加速不能改壞成敗判定。所以我寫了對照 bench，在同一門檻下逐任務比對兩個引擎的成功/失敗，兩邊完全一致才採用。實際在 Kaggle GPU 上跑時還抓到一個只在真硬體才會現形的 bug——`PreTrainedConfig.type` 是唯讀 property，舊程式碼手動賦值它，但這條路徑被 lazy import 包住，Mac 上的 mock 測試（57 個全綠）涵蓋不到，只有真的載模型才會觸發。修掉後實測穩態約 2× 加速，parity 全部一致。這也說明了為什麼 mock 測試綠了還不能算數：介面邊界以外的整合行為，終究得在真環境驗一次。

**範圍界線。** 全程是模擬加真資料，沒有做實體手臂的 closed-loop 評估——真機上「policy 輸出動作 → 手臂執行 → 相機回授 → 下一步」這個閉環尚未跨越，文件中明確標註，不含糊帶過。整條走過的路徑：編排層解耦 → 真資料微調 VLA → 執行管線優化，對應的是這個領域實際的工作流順序，而不是只挑一個點做深。

程式碼與設計拆解：<https://github.com/Kaminoikari/physical-ai-agent>

---

## 短版（Threads）

> VLA 模型已經能直接吃語言指令，為什麼還需要在上面疊一層 LLM？
>
> 因為 VLA 一次只做一個原子動作。組合多步、決定順序、判斷自己能不能做、指令有歧義時澄清——這些是編排層的職責，不是 policy 的。
>
> 這個 side project 把編排層做出來、驗證它與 policy 解耦：
> — 用 Claude 做指令拆解與排序，測的是邊界行為（拒絕、澄清、失敗後中止），不是成功展示
> — 用社群公開的 SO-101 真機資料在 Kaggle fine-tune SmolVLA，跑通資料到模型的流程
> — 執行管線從每任務重載 policy 改為常駐複用，對照測試確認成敗一致後採用，約 2× 加速
>
> 範圍：模擬加真資料，未做實體手臂 closed-loop。
>
> 模型：huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft
> 程式碼：github.com/Kaminoikari/physical-ai-agent
