# 軸 A：plan-only 拆解驗證設計

> 日期：2026-06-16
> 狀態：已核可，待實作

## 目標

放大本專案的核心論點「**為什麼具身智慧需要一層 LLM 編排大腦（L3）**」。前四／五個情境
證明了組合（「都」）、邊界、歧義、真實失敗；軸 A 再加上更難的**語言拆解**——語意分群、
排除／否定、3+ 物件、排序約束——來證明 L3 在「把人話翻成原子任務序列」這件事上做的，
是 VLA policy 完全做不到的世界知識與語言推理。

關鍵取捨：**只驗 agent 的拆解計畫（plan），不跑 rollout**。因此零 GPU、零 sim、秒回、
幾乎零成本（只花 Claude API），是最高面試 CP 值的擴充。

## 為什麼用獨立 CLI 而非在 `demo_libero.py` 加 flag

`demo_libero.py` 一開頭就 `LiberoSkillInterface(suite=...)`，其 `__init__` →
`_load_task_list()` → `import libero`，在 Mac 本機直接失敗。要在同檔加 `--plan-only`
就得在建構子前分叉、繞過 skills，把兩條路徑纏在一起。

改用**獨立 `demo_plan.py` + 純函式模組 `agent/plan_only.py`**：plan-only 路徑完全不依賴
lerobot，本機可跑、可單元測試；Kaggle 再用 `--live` 拉真選單。職責乾淨、邊界清楚。

## 架構

```
demo_plan.py (CLI)
   │  本機：靜態選單 LIBERO_OBJECT_TASKS（零 lerobot）
   │  --live：LiberoSkillInterface(suite).available_tasks()（Kaggle 真選單）
   ▼
agent/plan_only.py（純函式、可測）
   decompose_only(brain, instruction) -> Plan      # 就是 brain.decompose() 一次，不執行
   format_plan(plan, tasks) -> list[str]           # Plan → 人類可讀行；execute 標「計畫」、id 映回語言
   ▼
agent/brain.py（沿用）Brain.decompose → Plan
```

plan-only **完全不碰 `skills.execute()`**，不存在「成功／失敗」結果，只有「計畫」。

## 改動清單

### 1. `agent/libero_prompts.py`（改）
新增靜態選單常數，與真 LIBERO 一致：
```python
LIBERO_OBJECT_TASKS: list[tuple[int, str]] = [
    (0, "pick up the alphabet soup and place it in the basket"),
    ...
    (9, "pick up the orange juice and place it in the basket"),
]
```
本機 plan-only 用它建 system prompt，零 lerobot 依賴。

### 2. `agent/plan_only.py`（新，純函式）
- `decompose_only(brain, instruction) -> Plan`：呼叫 `brain.decompose(instruction)` 一次。
- `format_plan(plan, tasks) -> list[str]`：把 `Plan` 轉成可讀行。規則：
  - out_of_scope → 一行「⛔ 超出範圍：<reasoning>」
  - needs_clarification → 「❓ 需要澄清：<question>」
  - 否則逐 step：execute 標成「**計畫（未跑 rollout）** execute(task N: <language>)」，
    把 task_id 映回語言；**絕不出現「成功」字樣**（plan-only 不宣稱執行結果）。

### 3. `demo_plan.py`（新 CLI）
```bash
python demo_plan.py "把所有醬料都收進籃子"                              # 本機：靜態選單
python demo_plan.py "把所有醬料都收進籃子" --live --suite libero_object  # Kaggle：真選單
```
`--live` 才 import `LiberoSkillInterface`；不加就用靜態常數。`--model` 預設
`claude-sonnet-4-6`。

## 四情境與預期對映（驗收基準）

醬料 = salad dressing(2)、bbq sauce(3)、ketchup(4)、tomato sauce(5)。

| # | 指令 | 證明能力 | 預期 plan |
|---|---|---|---|
| 語意分群 | 把所有醬料都收進籃子 | 類別→具體物件的世界知識分類 | execute(2,3,4,5) |
| 排除/否定 | 除了番茄醬，其他醬料都收進籃子 | 分群 + 否定語法 | execute(2,3,4) |
| 3+ 物件 | 把字母湯、牛奶和柳橙汁都收起來 | 多步拆解可擴展到 3+、跨語言映射 | execute(0,7,9) |
| 排序約束 | 先收番茄醬再收字母湯 | 尊重顯式排序而非任意排 | execute(5)→execute(0) |

## 測試（全程 mock，零 API）

`tests/test_plan_only.py`，用 `FakeClient`：
1. `format_plan` 對 4 步 execute plan 標「計畫」且輸出**不含「成功」**。
2. `format_plan` 正確把 task_id 映回語言字串。
3. 語意分群指令 → FakeClient 回 4 步 plan → 格式化出 4 行 execute。
4. out_of_scope plan → 格式化出「超出範圍」、不含 execute 行。
5. needs_clarification plan → 格式化出澄清問句。

## 真實證據如何產

用真 Claude（`ANTHROPIC_API_KEY`）對四句指令各跑一次 `demo_plan.py`，蒐集真 log 寫進
`docs/demo-results.md` 新增「⑥ 語意拆解（plan-only，零 GPU）」段落，並同步 README 表格與
`docs/article-physical-ai-agent.md`。

**system prompt 預設不改**——重點正是「同一個 prompt、更難的語言，agent 仍拆對」。若某
情境真 Claude 拆錯，如實記錄（延續本專案誠實基調），必要時最小幅補 prompt 並註明。

## 風險與誠實限制

- **靜態選單是手動 snapshot**：與真 LIBERO 動態列出的選單一致，但若官方 suite 改版需手動
  同步。`--live` 模式用真 `available_tasks()` 可作為對照。
- **plan-only 不證明 rollout 會成功**：它只證明「拆解正確」這一層；執行層成敗另由情境
  ①〜⑤的真 rollout 證據負責。文件需講清楚這個邊界，不過度宣稱。
- **真 Claude 可能拆錯**：四情境屬非平凡語言推理，不保證一次到位；如實記錄結果。
