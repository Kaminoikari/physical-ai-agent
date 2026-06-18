# 學習記錄草稿（對外貼文）

> 連結：
> - GitHub repo：<https://github.com/Kaminoikari/physical-ai-agent>
> - fine-tune 的 VLA 模型：<https://huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft>

---

## 長版

### Physical AI 學習記錄：把編排層與 policy 解耦，並驗證它

這是一個關於 embodied AI 編排層的 side project。出發點是一個問題：VLA（Vision-Language-Action）模型已經能直接吃語言指令，為什麼還需要在它上面疊一層 LLM？

因為 VLA 的操作粒度是單一原子動作。組合多步、決定順序、判斷指令是否超出能力、在指令有歧義時澄清——這些屬於上層編排的職責，不是 policy 的職責。這個專案把這層（L3）實作出來，並驗證它與底層 policy（L1）解耦。

**架構。** 三層：L3 用 Claude 做指令拆解與排序；L2 是技能介面（`execute` / `query`）；L1 是 SmolVLA 在 LIBERO 模擬裡跑單一抓放。同一套編排骨架在 mock 與真模擬之間共用，換模擬環境只需替換 L2 的技能實作，L3 的迴圈、重試、中止邏輯不變。

**行為驗證。** 驗證重點放在邊界情況，而非成功展示：多步指令的拆解與排序、超出能力時拒絕執行、指令歧義時反問、長程任務（官方 checkpoint 在該 suite 實測約 5% 成功率）失敗後中止並回報原因。重試與中止路徑由單元測試固定，不依賴單次 rollout 的運氣。

**真機資料微調。** 為了補上「真實機器人資料 → policy」這一段，我用社群公開的 SO-101 遙操作資料集，在 Kaggle 上 fine-tune SmolVLA、產出 checkpoint 並上傳（[Kaminoikari/smolvla-so101-pickplace-ft](https://huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft)）。這一步確認的是整條資料到模型的流程可跑通，使用公開資料、不涉及實體硬體。

**執行管線。** 原本每個任務都透過 subprocess 重載一次 policy。我把它改為 policy 與 env 常駐複用、直接呼叫 `eval_policy()`，並以對照測試確認新舊路徑在同一門檻下成敗判定一致後才採用。實測穩態約 2× 加速。

**範圍界線。** 全程為模擬加真資料，未做實體手臂的 closed-loop 評估——這部分尚未跨越，文件中明確標註。整條路徑：編排層 → 真資料微調 VLA → 執行管線優化，對應這個領域的實際工作流。

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
