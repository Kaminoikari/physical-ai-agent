# Demo 結果速查（面試用）

> 一句話：**Claude 當 L3 編排大腦，把自然語言指令拆解／排序成 LIBERO 任務序列，
> 交給官方 SmolVLA（L1）在真模擬裡執行，並以 ground-truth success 閉環。**
> 以下①〜⑤全部在 Kaggle T4 上、用官方 `HuggingFaceVLA/smolvla_libero` checkpoint
> 跑 `lerobot-eval` 真 rollout 產出（非 mock、非 assume_success）。①〜④在 `libero_object`
> suite（短任務），⑤在 `libero_10` 長程 suite（真實失敗素材）。
> ⑥是 plan-only 拆解驗證：只跑真 Claude 拆解、**不跑 rollout**，本機零 GPU、秒回，專測更難的
> 語言推理（語意分群／排除／3+ 物件／排序），並揭露「番茄醬」這類真歧義。

## 三層架構

```
L3  Agent 編排（Claude sonnet-4-6）   理解人話 → 拆解 → 排序 → 邊界/歧義判斷
        │  agent/agent.py（mock 與真 sim 共用同一套編排骨架）
        ▼
L2  技能介面（SkillInterface）          execute(task_id) / query
        │  agent/libero_skills.py（真版）｜ agent/skills.py（mock 版）
        ▼
L1  Policy 執行（SmolVLA + LIBERO）     單一原子任務的抓放 rollout
```

關鍵設計：**L1 完全解耦**。本機用 MockWorld，雲端換成真 LIBERO，**L3 的編排邏輯
（拆解迴圈／重試／abort／邊界判斷）不變**；換 sim 主要動 L2 技能實作（`skills.py` →
`libero_skills.py`），`agent.py` 僅為 task-level `execute` 加一個動作分派分支與誠實的失敗措辭。
（這比一句「完全不改」更精準——技能顆粒度從 pick/place 變成 task-level 時，介面語意確實變了。）

## 任務選單（libero_object suite，agent 的技能清單）

agent 啟動時動態列出，prompt 不寫死任何任務：

```
0: pick up the alphabet soup and place it in the basket
1: pick up the cream cheese and place it in the basket
2: pick up the salad dressing and place it in the basket
3: pick up the bbq sauce and place it in the basket
4: pick up the ketchup and place it in the basket
5: pick up the tomato sauce and place it in the basket
6: pick up the butter and place it in the basket
7: pick up the milk and place it in the basket
8: pick up the chocolate pudding and place it in the basket
9: pick up the orange juice and place it in the basket
```

## 五種行為（核心證據鏈）

### ① 成功 — 語意對應 + 真閉環
指令：`把字母湯罐頭收進籃子`
```
• 拆解：使用者要求將字母湯罐頭放進籃子，對應任務 0：pick up the alphabet soup and place it in the basket
• execute(task 0: pick up the alphabet soup and place it in the basket) -> 成功
✅ 完成：任務完成
```
**證明**：中文指令 → 英文任務的語意對應正確；execute 回傳的是 LIBERO ground-truth success，不是假設成功。

### ② 多步 — 任務分解 + 排序
指令：`把字母湯罐頭和番茄醬都收進籃子`
```
• 拆解：使用者要求將字母湯（alphabet soup，task 0）和番茄醬（tomato sauce，task 5）都放入籃子，
        兩者皆在可用任務清單內，依序執行即可。
• execute(task 0: pick up the alphabet soup and place it in the basket) -> 成功
• execute(task 5: pick up the tomato sauce and place it in the basket) -> 成功
✅ 完成：任務完成
```
**證明**：「都」一個字 → agent 拆成兩個 execute 依序執行，各自跑真 rollout 並回 success。
（誠實註記：此處 agent 把「番茄醬」對到 tomato sauce(5)，但繁中口語「番茄醬」其實偏指
ketchup——同一個詞在情境⑥的 plan-only 裡又一致對到 ketchup(4)。這是**真歧義**、不是「對 5 才正確」，
詳見情境⑥。本段重點在「都」的組合/排序，不在番茄醬的指代。）
**這是「為什麼還需要 L3」的最佳反例**：SmolVLA 一次只吃一個原子任務，組合/排序是 L3 的活。

### ③ 拒絕 — 能力邊界自覺
指令：`幫我把桌子擦乾淨`
```
• 拆解：使用者要求擦桌子，這個動作不在任何已訓練任務的範圍內，
        所有可用任務均為抓取物品並放入籃子，無法對應此需求
⛔ 超出任務範圍：…
```
**證明**：在「動手前」的 reasoning 階段就判斷出超出能力，**完全沒跑 rollout**——
不是抓了一半才失敗，而是知道自己不會就直說（in_scope: false）。

### ④ 澄清 — 不幻覺、會反問
指令：`把那個東西收起來`
```
• 拆解：使用者說「那個東西」，但沒有指明是哪樣物品，無法對應到任何具體任務，需要先釐清。
❓ 需要澄清：請問您想收起來的是哪樣東西？例如：alphabet soup、cream cheese、…、orange juice？
```
**證明**：指代不明時不亂猜，反問並列出可選項（needs_clarification: true）。

### ⑤ 真實失敗 — 重試上限 + 誠實 abort（libero_10 長程任務，真 L1 失敗）
指令：`put both the alphabet soup and the tomato sauce in the basket`（`--suite libero_10`）
```
• 拆解：使用者要求將 alphabet soup 和 tomato sauce 都放入籃子，直接對應任務 0。
• execute(task 0: put both the alphabet soup and the tomato sauce in the basket) -> 失敗
• rollout 失敗，重試 execute（第 1 次）
• execute(task 0: put both the alphabet soup and the tomato sauce in the basket) -> 失敗
• rollout 失敗，重試 execute（第 2 次）
• execute(task 0: put both the alphabet soup and the tomato sauce in the basket) -> 失敗
🛑 中止：execute(0) 連續失敗
```
**證明**：`libero_10` 是長程任務（單一 rollout 內要依序收兩樣物品），官方 checkpoint 在此 task
**實測 ~5% 成功率**（首次 probe 5 集估出 20%，但加計後續 4 次 demo 共 20 次 rollout 只成功 1
次——單次小樣本高估了）。agent 每次都跑**真** rollout、各自獨立擲骰，全部失敗後誠實 abort 並
講明原因。重試措辭是「**rollout 失敗**」（task-level 沒有座標驗證那步，如實反映真正失敗的環節），
不是含糊的「驗證失敗」。**這是把 mock `--fail-first` 假失敗換成真 L1 失敗**：失敗訊號來自
LIBERO 的 ground-truth `pc_success`，不是腳本注入——「失敗當一等公民」從假證據變真證據。
> 註：「失敗→重試→成功」的救回結局在程式上存在、且有確定性單元測試
> `test_execute_failure_then_retry_succeeds` 證明；但這顆 checkpoint 在此 task 太弱（~5%），
> 跑了 4 輪、15 次真 rollout 一次都沒救回（含 `--max-retries 5` 的 6 連敗）——**真實世界觀察到
> 的結局一致是誠實 abort**。重試邏輯靠單元測試保證，不靠僥倖拍到一支救回影片。每次 rollout
> 影片（含失敗畫面）持久保存在 `/kaggle/working/libero_exec/<suite>/task<id>/run<n>/`。

## ⑥ 語意拆解 — plan-only，零 GPU（語意分群／排除／3+ 物件／排序）

前五種都跑真 rollout（要 GPU）。第六種把「**拆解人話 → 任務序列**」這層單獨拎出來驗證——
只呼叫 `brain.decompose()` 一次、印出計畫，**不跑 rollout**，本機秒回、只花 Claude API。
重點正是「同一個 system prompt、更難的語言，agent 仍拆得出結構化計畫」。以下四段是真 Claude
（`claude-sonnet-4-6`）對 `libero_object` 選單的實際輸出，用 `python demo_plan.py "<指令>"` 產出。

**⑥-a 語意分群 — 類別→具體物件的世界知識**
指令：`把所有醬料都收進籃子`
```
拆解：使用者要求將所有醬料收進籃子。選單中屬於醬料類的品項包括：salad dressing (2)、
      bbq sauce (3)、ketchup (4)、tomato sauce (5)，依序執行這四個任務。
📋 計畫（未跑 rollout，只驗拆解）：
  1. execute(task 2: pick up the salad dressing and place it in the basket)
  2. execute(task 3: pick up the bbq sauce and place it in the basket)
  3. execute(task 4: pick up the ketchup and place it in the basket)
  4. execute(task 5: pick up the tomato sauce and place it in the basket)
```
**證明**：「醬料」這個**類別**沒出現在任何 task 字面上，agent 得用世界知識把 10 項物件裡的 4 項
醬料挑出來。這是 VLA policy 完全做不到的——它只吃「pick up the X」單一指令，不懂「所有醬料」。

**⑥-b 排除／否定 + 真歧義（番茄醬）**
指令：`除了番茄醬，其他醬料都收進籃子`
```
拆解：使用者要求將除了番茄醬（ketchup, task 4）以外的所有醬料收進籃子。從選單中屬於醬料類的有：
      cream cheese(1)、salad dressing(2)、bbq sauce(3)、ketchup(4)、tomato sauce(5)。
      排除 ketchup 後，需執行 task 1、2、3、5。
📋 計畫（未跑 rollout，只驗拆解）：
  1. execute(task 1: pick up the cream cheese and place it in the basket)
  2. execute(task 2: pick up the salad dressing and place it in the basket)
  3. execute(task 3: pick up the bbq sauce and place it in the basket)
  4. execute(task 5: pick up the tomato sauce and place it in the basket)
```
**證明 + 兩個誠實發現**：否定語法本身判對了（「除了 X」→ 從集合移除 X）。但這一題暴露兩個真實的
語言坑，比「全對」更有教材價值：
- **`番茄醬` 是真歧義**：選單同時有 `ketchup`(4) 和 `tomato sauce`(5)。繁中口語「番茄醬」通常指
  ketchup（紅色那條），義式紅醬才是 tomato sauce。這裡 agent 把番茄醬判成 **ketchup(4)** 並排除——
  語言上站得住。但情境②（真 rollout）裡同一個詞卻對到 **tomato sauce(5)**。**同一個 agent、同一個詞、
  不同情境給出不同對映**——這正說明它是該觸發澄清的歧義，沒有單一「正確答案」。
- **類別邊界模糊**：agent 把 `cream cheese`(1, 奶油乳酪) 也算進「醬料」，⑥-a 卻沒算。類別歸屬本身
  浮動。這不是 bug，是自然語言固有的模糊，恰恰是 L3 該面對、policy 碰不到的問題。

**⑥-c 3+ 物件明列 — 多步拆解可擴展**
指令：`把字母湯、牛奶和柳橙汁都收起來`
```
拆解：使用者要求將字母湯（task 0）、牛奶（task 7）和柳橙汁（task 9）依序放入籃子，
      三個任務均在選單內，依序執行即可。
📋 計畫（未跑 rollout，只驗拆解）：
  1. execute(task 0: pick up the alphabet soup and place it in the basket)
  2. execute(task 7: pick up the milk and place it in the basket)
  3. execute(task 9: pick up the orange juice and place it in the basket)
```
**證明**：三個中文物件名跨語言正確對到 0/7/9，且依序排成 3 步。比情境②的兩步「都」更進一步，
證明多步拆解可擴展、不是只能處理兩樣。

**⑥-d 排序約束 — 尊重顯式順序**
指令：`先收番茄醬再收字母湯`
```
拆解：使用者要求依序收番茄醬（ketchup, task_id=4）再收字母湯（alphabet soup, task_id=0）
📋 計畫（未跑 rollout，只驗拆解）：
  1. execute(task 4: pick up the ketchup and place it in the basket)
  2. execute(task 0: pick up the alphabet soup and place it in the basket)
```
**證明**：「先 A 再 B」→ 計畫順序是 A(4) 在前、B(0) 在後，agent 尊重顯式排序而非任意排。
（番茄醬同樣對到 ketchup(4)，與 ⑥-b 一致——這個 agent 對「番茄醬」的偏好是穩定的 ketchup。）

**⑥-e `--live` 雲端交叉驗證（Kaggle，真 `available_tasks()` 選單）**
上述四段用本機靜態 snapshot 跑；在 Kaggle 用 `--live` 取真 LIBERO 動態選單重跑一次，確認三件事：
- **選單一字不差**：`--live` 印出的真 `available_tasks()` 選單 0〜9 與靜態 `LIBERO_OBJECT_TASKS`
  完全一致（`check_menu.py` 驗 ✅）。本機 plan-only 的 task_id 對映可信、非手寫偏差。
- **番茄醬→ketchup(4) 在真環境也穩**：②④兩處真選單下同樣對到 ketchup(4)，跨「本機/雲端 × ②/④」
  四處一致——歧義是 agent 對「番茄醬」的真實偏好，不是本機產物。
- **類別歸屬會跳（加碼證據）**：①「所有醬料」本機跑成 `execute(2,3,4,5)`（未含 cream cheese），
  `--live` 重跑卻成 `execute(1,2,3,4,5)`（**含** cream cheese (1)）。同一句、同一選單，cream cheese
  算不算「醬料」在不同次之間翻覆——比單次的「①沒算、②算」更鐵地證明類別邊界本身浮動，正是
  該觸發澄清、policy 碰不到的語言坑。

> 邊界誠實：plan-only 只證明「**拆解這一層**」對；它不證明 rollout 會成功（那由①〜⑤的真 rollout
> 負責）。兩層分明、不混為一談。靜態 snapshot 與真 sim 選單一致已由 `--live` + `check_menu.py` 確認。

## ⑦ rollout 加速實測（in-process vs subprocess）

方法：`bench_rollout.py` 對同批 task 跑兩引擎，量每 task 牆鐘時間並斷言成敗 parity（兩引擎同門檻下成敗判定一致）。
重點：`InProcessRolloutEngine` 把 policy/env 常駐複用，消除 subprocess 版「每 task 重載 policy」的開銷；
預期每 task 從 ~3 min（subprocess 重載）降到數秒級。成敗判定邏輯不變（門檻仍套用 `pc_success`），故可信度不動。
> 實測數字待在 Kaggle／租用 GPU 跑出後填入（含 SmolVLA；GR00T N1.5 視 Ampere+ GPU 取得情況，見 spec 附錄 A）。

## 如何重現

完整步驟見 `docs/week2-kaggle-agent-libero-run.md` 與 `docs/week2-kaggle-libero-setup.md`。
摘要：Kaggle 開 GPU T4 + Internet → 裝 `lerobot[smolvla,libero]` → 寫 `~/.libero/config.yaml`
→ Kaggle Secrets 載 `ANTHROPIC_API_KEY`/`GITHUB_PAT` → clone repo →
`python demo_libero.py "<指令>" [--suite libero_10]`（預設 `libero_object`；真實失敗情境用 `libero_10`）。

本機（Mac）跑 mock 版同樣四情境（零 API 以外成本、秒回）：
`.venv/bin/python demo.py "<指令>" [--fail-first 1]`（`--fail-first` 可現場展示失敗→重試→成功）。

情境⑥（plan-only 拆解驗證）本機即可跑，零 GPU、只花 Claude API：
`.venv/bin/python demo_plan.py "把所有醬料都收進籃子"`（本機靜態選單）；
Kaggle 上加 `--live` 改用真 `available_tasks()` 選單作對照。

## 誠實的限制

- **慢（已著手解）**：subprocess 版每次重載 policy（~3 min/task）。已實作 `InProcessRolloutEngine`（policy/env
  常駐複用，直接呼叫 `eval_policy()`）解此瓶頸，subprocess 版保留為 parity baseline；實測加速數字待 Kaggle 填（見 ⑦）。
- **成功率受限於官方 checkpoint，未自行微調**：`libero_object` 單物件抓放 100%；`libero_10`
  長程任務（單 rollout 多步）官方 checkpoint 實測僅 ~5%——夠當「真實失敗」素材，但不適合當 happy-path。
  未涵蓋雙臂/堆疊。要拉高長程成功率得自行微調（本專案刻意只用官方 checkpoint、不微調）。
- **Week 1 用 metaworld expert 而非 SmolVLA**：SmolVLA 的 SO101 action/obs space 與 metaworld 4-dim
  不相容，zero-shot 接不上；Week 1 的「看到手臂動」是用 metaworld 內建 expert 腳本驅動。
  真正用 SmolVLA 驅動是在 Week 2 的 LIBERO（已對齊 action space 的官方 checkpoint）。
- **sim-to-real gap**：全程純模擬，未上實體手臂。
