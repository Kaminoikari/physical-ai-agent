# LIBERO × Agent 編排層整合設計（task-level 技能）

> 日期：2026-06-15 ｜ 對應 spec §5 Week 2→3 銜接、§9 升級路徑 ｜ 狀態：已選方案 A，待 user 過目
> 前置：mock-world agent 層已完成（`agent/`，29 測試綠）；Kaggle 上 LIBERO + 官方
> `HuggingFaceVLA/smolvla_libero` eval 已通（libero_object task0 = 100% success）。

## 目標
把已驗證的 L3 編排層（Claude 拆解 + 閉環重試）接上**真 LIBERO + 真 SmolVLA**，
做到「一句人話 → agent 調度預訓練 task policy → 真 sim 執行 → 用成功訊號閉環」。

## 關鍵架構決定（方案 A：task-level 技能）
SmolVLA+LIBERO 的最小執行單位是**整段 task 的語言條件式 rollout**（無「只 pick」子技能）。
因此 L2 技能粒度從 mock 的 `pick/place` 改為 **task-level**：

- 技能 = `execute(task)`：在 LIBERO env 跑一次 SmolVLA rollout，回傳該 task 的成功與否。
- agent 的工作 = 把使用者人話**對應/排序**成 LIBERO 任務指令，並用成功訊號決定重試/放棄。

> 這正是 spec 核心命題「LLM 調度有限的預訓練 policy」的真實版（NVIDIA 架構），
> pick/place 粒度只是 mock 的簡化。L3 編排骨架（範圍判斷、澄清、序列、閉環重試）全保留。

## 架構（沿用三層，換 L1/L2 實作）
```
agent/
  libero_skills.py   # 新：LiberoSkillInterface（task-level，取代 MockWorld+SkillInterface）
  libero_prompts.py  # 新：task-level 系統 prompt（技能字彙=execute + 可用任務清單）
  agent.py           # 沿用：編排迴圈（decompose→execute→verify→retry/abort）幾乎不動
  brain.py/schemas.py# 沿用
demo_libero.py       # 新：Kaggle 上的端到端 demo 入口
```

### L1+L2 — LiberoSkillInterface（`libero_skills.py`）
以 lerobot 既有 API 為底（已查證）：
- `available_tasks(suite) -> list[(task_id, language)]`：用 `benchmark.get_benchmark_dict()` 取
  suite、`task_suite.get_task(i).language` 列出每個任務的自然語言指令（= agent 的技能選單）。
- `execute(task_id) -> SkillResult`：建 `LiberoEnv`（該 task）、用已載入的 `smolvla_libero`
  policy 跑 rollout（重用 `lerobot.scripts.lerobot_eval.rollout`），回傳 `info["is_success"]`。
- `query(task_id, mode="spatial") -> bool`：回該 task 的成功訊號（ground-truth），供閉環驗證。
- policy 與 benchmark suite **載入一次、跨 execute 重用**（避免每次重載 450M 權重）。

### L3 — prompt 調整（`libero_prompts.py`）
- 技能字彙：`execute(task)`、`query`；附上 `available_tasks()` 的清單讓 agent 從中選。
- 規則沿用：超範圍→`in_scope:false`；指令對不到任何任務或有歧義→`needs_clarification`；
  多目標→輸出 execute 序列。輸出仍是既有 JSON 格式（`plan` 內 skill 改為 execute/query）。

### L3 — 迴圈（`agent.py`，沿用）
decompose→逐步 execute→以 success 驗證→失敗重試（上限 2）→再失敗 abort。
與 mock 版唯一差異是「skill 執行器」換成 LiberoSkillInterface、技能名是 execute。
（v1 仍不做動態重規劃。）

## 執行環境
**全部跑在 Kaggle**：`agent/` 程式 + Claude API（Kaggle 有 Internet）+ LIBERO rollout（T4 GPU）同機。
務實限制：每個 task rollout 在 T4 約 3 分鐘，agent 多步/重試是「分鐘級」，能 demo 但慢。

## 測試策略（誠實面對：LIBERO 無法在本機 Mac 跑）
- **本機（Mac）**：用 `FakeLiberoSkill`（in-memory，回腳本化 success）+ 既有 `FakeClient`，
  對 `agent.py` 的 task-level 編排做單元測試（decompose→execute→retry/abort 路徑），延續 TDD。
- **Kaggle（真實）**：手動跑 `demo_libero.py` 做整合驗證（真 rollout + 真 success）。
- 不在 CI 跑 LIBERO（需 Linux+GPU）。

## 驗收標準
- 本機：task-level 編排單元測試全綠（含 execute 成功序列、重試、abort、in_scope、needs_clarification）。
- Kaggle：`demo_libero.py "把字母湯罐頭收進籃子"` → agent 對應到正確 LIBERO 任務 → 真 rollout
  成功 → 印出拆解+執行+成功訊號。能展示一次「多任務序列」與一次「失敗重試」。

## 風險與對策
- **rollout 慢（T4 ~3min/task）** → demo 用最少 episode、單一 suite；錄影後加速播放。
- **人話↔任務對應不準** → 把 available_tasks 的 language 清單放進 prompt，讓 agent 從固定選單選，
  降低幻覺；對不到就 needs_clarification。
- **Kaggle session 逾時/不持久** → 把 `agent/` 與 demo 放 GitHub，notebook 開頭 git clone + 裝。
- **lerobot rollout API 細節變動** → 以查證過的 0.5.x（`rollout()`、`task.language`、`info["is_success"]`）為準；實作時若簽章不符，依當下原始碼微調。

## 不做（YAGNI）
動態重規劃、自己微調 policy、pick/place 子技能切分、把 agent 跑在本機而 LIBERO 在雲端的 RPC 架構。
