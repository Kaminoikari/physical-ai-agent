# Demo 結果速查（面試用）

> 一句話：**Claude 當 L3 編排大腦，把自然語言指令拆解／排序成 LIBERO 任務序列，
> 交給官方 SmolVLA（L1）在真模擬裡執行，並以 ground-truth success 閉環。**
> 以下五段 log 全部在 Kaggle T4 上、用官方 `HuggingFaceVLA/smolvla_libero` checkpoint
> 跑 `lerobot-eval` 真 rollout 產出（非 mock、非 assume_success）。前四段在 `libero_object`
> suite（短任務），第五段在 `libero_10` 長程 suite（真實失敗素材）。

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
番茄醬正確對到 tomato sauce(5) 而非 ketchup(4)，跨語言歧義判對。
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
實測 `pc_success` 僅 **~20%**（單獨 probe：5 集中僅 1 集成功）。agent 跑滿 3 次**真** rollout
（每次都是獨立的 20% 擲骰）全部失敗後，誠實 abort 並講明原因。重試措辭是「**rollout 失敗**」
（task-level 沒有座標驗證那步，措辭如實反映真正失敗的環節），不是含糊的「驗證失敗」。
**這是把 mock `--fail-first` 假失敗換成真 L1 失敗**：失敗訊號來自 LIBERO 的 ground-truth
`pc_success`，不是腳本注入——「失敗當一等公民」從假證據變真證據。
> 註：~20% 任務的主流結局是 abort（每輪約 51%）；當某次 rollout 偶然成功時，也會出現
> 「失敗→重試→成功→✅」的救回結局（每輪約 49%）。兩種結局都純由真 rollout 結果驅動，
> agent 不靠 assume_success。每次 rollout 影片（含失敗畫面）持久保存在
> `/kaggle/working/libero_exec/<suite>/task<id>/run<n>/`。

## 如何重現

完整步驟見 `docs/week2-kaggle-agent-libero-run.md` 與 `docs/week2-kaggle-libero-setup.md`。
摘要：Kaggle 開 GPU T4 + Internet → 裝 `lerobot[smolvla,libero]` → 寫 `~/.libero/config.yaml`
→ Kaggle Secrets 載 `ANTHROPIC_API_KEY`/`GITHUB_PAT` → clone repo →
`python demo_libero.py "<指令>" [--suite libero_10]`（預設 `libero_object`；真實失敗情境用 `libero_10`）。

本機（Mac）跑 mock 版同樣四情境（零 API 以外成本、秒回）：
`.venv/bin/python demo.py "<指令>" [--fail-first 1]`（`--fail-first` 可現場展示失敗→重試→成功）。

## 誠實的限制

- **慢**：`libero_skills.py` 的 execute 以 subprocess 呼叫 `lerobot-eval`，每次重載 policy（~3 min/task）。
  取捨是「重用整條官方管線、最穩」；可改為「載 policy 一次 + 直接呼叫 `eval_policy()`」加速。
- **成功率受限於官方 checkpoint，未自行微調**：`libero_object` 單物件抓放 100%；`libero_10`
  長程任務（單 rollout 多步）官方 checkpoint 僅 ~20%——夠當「真實失敗」素材，但不適合當 happy-path。
  未涵蓋雙臂/堆疊。要拉高長程成功率得自行微調（本專案刻意只用官方 checkpoint、不微調）。
- **Week 1 用 metaworld expert 而非 SmolVLA**：SmolVLA 的 SO101 action/obs space 與 metaworld 4-dim
  不相容，zero-shot 接不上；Week 1 的「看到手臂動」是用 metaworld 內建 expert 腳本驅動。
  真正用 SmolVLA 驅動是在 Week 2 的 LIBERO（已對齊 action space 的官方 checkpoint）。
- **sim-to-real gap**：全程純模擬，未上實體手臂。
