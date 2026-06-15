# Agent 編排層設計（L3/L2 + Mock L1）

> 日期：2026-06-15 ｜ 對應 spec §5 Week 3–4 ｜ 狀態：已與使用者確認，待轉實作計畫

## 目標

在本機（無 GPU、無真實 sim）把 Physical AI agent 的**最高價值層**做出來並驗收：
使用者用一句沒寫死流程的自然語言指令 → agent（Claude）拆解成 `pick`/`place`/`query`
技能序列 → 在 mock world 上執行 → 用座標真值做閉環回饋 → 失敗時有限重試、必要時放棄。

底層 L1 用 **MockWorld** 當 SmolVLA/LIBERO 的暫時替身。L1 與 L2/L3 解耦，未來雲端接
LIBERO 只替換 `world.py`/`skills.py`，L3 不動。這直接驗證 spec 的「L1 解耦」設計鐵則。

## 範圍

**In scope**：mock world、三技能介面、Claude 拆解、端到端編排迴圈（assume-success 與
全閉環兩模式）、座標回饋的重試/abort、agent 文件 5 個測試案例驗收、CLI demo。

**Out of scope（YAGNI）**：依回饋動態重新規劃（v1 只做「整段拆解 + 逐步驗證/重試」）、
真實物理、真實 VLM、多於 pick/place/query 的技能。

## 架構

```
agent/
  world.py     # L1 替身 MockWorld
  skills.py    # L2 技能介面 pick/place/query
  brain.py     # L3 大腦 LLMClient 介面 + AnthropicClient + FakeClient
  prompts.py   # 系統 prompt（搬 agent-task-decomposition-prompt.md）+ 回饋模板
  schemas.py   # Plan/Step/SkillResult 型別（dataclass）
  agent.py     # L3 編排迴圈
demo.py        # CLI demo 入口
tests/         # test_world / test_decomposition / test_agent_loop
```

### L1 — MockWorld（world.py）
- 物件 `{id, color, pos:(x,y), held:bool}`；zones `{name: (x_min,x_max,y_min,y_max)}`；夾爪持有狀態。
- 初始場景對齊 demo 案例：`red_cube`、`blue_cube`、zones `zone_A`、`box_left`、`box_right`。
- `pick(query)`：依顏色/id 找物件，成功則設 held。`place(target)`：把持有物件移入目標 zone 範圍內、釋放。
- 真值查詢：`is_in_zone(obj, zone) -> bool`、`get_position(obj)`、`list_objects()`。
- **失敗注入**：`fail_next_pick` / 腳本化結果，供測試與 demo 重現「抓取失敗 → 重試」。

### L2 — 技能介面（skills.py）
- 只暴露三個 agent 可呼叫技能；簽章固定，是 L1 解耦邊界。
- `pick(object:str) -> SkillResult`、`place(target:str) -> SkillResult`。
- `query(question:str, mode:"spatial"|"semantic")`：
  - `spatial` → 讀 world 座標真值（100% 準），回 bool。
  - `semantic` → mock 階段由 world 狀態確定性作答；真實階段換 VLM，介面不變。

### L3 — 大腦（brain.py / prompts.py）
- `LLMClient` protocol：`complete(system:str, messages:list) -> str`。
- `AnthropicClient`：真實，預設 `claude-sonnet-4-6`，可設定。API key 由 `os.environ`
  讀取（`.env` 已 gitignore，絕不進版控）。
- `FakeClient`：回罐頭 JSON，單元測試零 API 費。
- `decompose(instruction, observation) -> ParsedPlan`：組 prompt → 呼叫 → 解析 JSON
  （`reasoning/in_scope/needs_clarification/clarification_question/plan[]`），含防呆 parse。

### L3 — 編排迴圈（agent.py）
1. `decompose` 得 plan。`in_scope:false` → 回報「超出技能範圍」停；
   `needs_clarification:true` → 回報 clarification_question 給使用者停。
2. 逐步執行：
   - `query` 步驟：執行後把結果回灌再 `decompose`（支援「先觀察再行動」，迭代上限防無限迴圈）。
   - `pick`/`place` 步驟：執行後 `query(mode="spatial")` 驗證 → 成功續、失敗重試（上限 2）、
     再失敗以 `abort(原因)` 收尾。
3. 兩模式：`assume_success=True` 跳驗證跑通端到端（Week 3）；`False` 開全閉環（Week 4）。

## 錯誤處理（系統邊界必處理）
- API 呼叫：try/except，限流/暫時錯誤小退避重試；最終失敗回明確錯誤。
- JSON 解析失敗：拋 `ParseError`、記原始回應，agent 回報失敗不崩潰。
- plan 出現未知技能 → abort。物件找不到 → 技能回失敗（驅動重試/abort）。
- 重試上限 2 強制；abort 路徑有測試覆蓋。

## 測試（TDD，先寫 failing test）
- `test_world.py`：pick/place/is_in_zone 機制與失敗注入。
- `test_decomposition.py`：agent 文件 5 案例（基本拆解 / 歧義→needs_clarification /
  未寫死規則一次排完 / 超範圍→in_scope:false / 疊放精度警示），用 FakeClient 驗
  parser 與路由；另附可選真 API smoke test（env 開關）驗 Claude 真的拆得對。
- `test_agent_loop.py`：注入抓取失敗 → 斷言重試後成功；持續失敗 → 斷言 2 次後 abort。

## 驗收標準
- 5 個拆解案例全綠（FakeClient）。
- 閉環測試：能重現一次「失敗 → 重試成功」與一次「持續失敗 → abort」。
- `python demo.py "把紅色方塊放到 A 區"` 印出 reasoning + plan + 逐步執行驗證 + 結果。
- 對齊 spec §6 指標：拆解正確率、端到端完成率、失敗重試可 demo。

## 安全備註
- ⚠️ 開發用的 Anthropic API key 曾在對話明文出現，視為已曝光；專案收尾前須至 console 撤銷重發。
- key 僅存於 gitignore 的 `.env`，程式以 `os.environ` 讀取，絕不 hardcode 或 commit。
