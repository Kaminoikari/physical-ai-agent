# physical-ai-agent

> 用 LLM（Claude）當具身機器手臂的**編排大腦**：把一句自然語言指令，拆解／排序成
> 一串「已訓練好的原子技能」，交給 VLA policy（SmolVLA）在模擬裡執行，並以
> ground-truth 成敗閉環。純模擬 side project。

一句話定位：**VLA 已經會吃語言了，為什麼還需要一層 LLM agent？** 因為 VLA 一次只
做得了一個原子任務，「把兩樣東西都收好」「擦桌子（我不會）」「把那個收起來（哪個？）」
這種**組合、排序、邊界判斷、歧義澄清**，是上層編排大腦（L3）的活。本專案就是把這層
做出來、並證明它與底層 policy 解耦。

## 三層架構

```
L3  Agent 編排（Claude sonnet-4-6）   理解人話 → 拆解 → 排序 → 邊界/歧義判斷
        │  agent/agent.py（mock 與真 sim 共用同一套編排骨架）
        ▼
L2  技能介面（SkillInterface）          execute(task_id) / query
        │  agent/libero_skills.py（真版：驅動 LIBERO+SmolVLA）
        │  agent/skills.py（mock 版：MockWorld）
        ▼
L1  Policy 執行（SmolVLA + LIBERO sim） 單一原子任務的抓放 rollout
```

**核心設計：L1 完全解耦。** 本機用 `MockWorld`，雲端換成真 LIBERO，**L3 的編排邏輯
（迴圈／重試／abort／邊界判斷）不變**——換 sim 主要動 L2 技能實作，`agent.py` 僅為
task-level `execute` 加一個動作分派分支。這是整個專案要證明的工程主張。

## 五種 agent 行為（全部在真 LIBERO 上跑出）

| 情境 | 指令 | agent 行為 | 證明的能力 |
|---|---|---|---|
| 成功 | 把字母湯罐頭收進籃子 | `execute(0)` → ✅ | 語意對應 + 真閉環 |
| 多步 | 字母湯和番茄醬**都**收進籃子 | `execute(0)`→`execute(5)` → ✅ | 任務分解 + 排序 |
| 拒絕 | 幫我把桌子擦乾淨 | ⛔ 超出範圍（不跑 rollout） | 能力邊界自覺 |
| 澄清 | 把**那個東西**收起來 | ❓ 反問 + 列選項 | 不幻覺、會反問 |
| 真實失敗 | （`libero_10`）兩樣都收進籃子 | 真 rollout 失敗×3 → 🛑 誠實 abort | 失敗當一等公民（真證據） |

前四種在 `libero_object`；第五種在 `libero_10` 長程任務（官方 checkpoint 僅 ~20% 成功率，
失敗訊號來自真 ground-truth、非 mock `--fail-first`）。

完整 log 與設計拆解見 [`docs/demo-results.md`](docs/demo-results.md)。
深入文章見 [`docs/article-physical-ai-agent.md`](docs/article-physical-ai-agent.md)。

## 快速開始

### 本機（Mac）跑 mock 版 — 秒回、零 sim 成本

需要 `ANTHROPIC_API_KEY`（放在 gitignore 的 `.env`）。

```bash
.venv/bin/python demo.py "把紅色方塊放到 A 區"
.venv/bin/python demo.py "把紅左藍右排好"          # 多物件
.venv/bin/python demo.py "幫我泡杯咖啡"            # 超出範圍
.venv/bin/python demo.py "把方塊放好" --fail-first 1  # 示範失敗→重試→成功
```

### 雲端（Kaggle GPU）跑真版 — 真 LIBERO + 官方 SmolVLA

完整步驟見 [`docs/week2-kaggle-libero-setup.md`](docs/week2-kaggle-libero-setup.md)
（環境）與 [`docs/week2-kaggle-agent-libero-run.md`](docs/week2-kaggle-agent-libero-run.md)
（跑 agent×LIBERO）。摘要：Kaggle 開 GPU T4 + Internet → 裝
`lerobot[smolvla,libero]` → 寫 `~/.libero/config.yaml` → Kaggle Secrets 載
`ANTHROPIC_API_KEY` → clone 本 repo → `python demo_libero.py "<指令>"`。

### 跑測試

```bash
.venv/bin/python -m pytest          # 35 tests，全程 mock，零 API 成本
```

## Repo 結構

```
agent/
  agent.py          L3 編排迴圈：decompose → 執行 → 驗證/重試 → abort（mock 與真共用）
  brain.py          LLMClient 介面 + AnthropicClient/FakeClient + parse_plan
  schemas.py        SkillResult / Step / Plan
  skills.py         L2 mock 技能（pick/place/query），驅動 MockWorld
  world.py          MockWorld：L1 替身（物件/區域/座標真值）
  prompts.py        mock 系統 prompt
  libero_skills.py  L2 真技能（execute/query/available_tasks），subprocess 驅動 lerobot-eval
  libero_prompts.py 真版系統 prompt（含動態任務選單）
demo.py             本機 mock CLI
demo_libero.py      Kaggle 真 LIBERO CLI
tests/              35 tests（TDD，全程 mock）
docs/               設計 spec、Kaggle 指南、demo 結果、文章
week1_*.py          Week 1：metaworld 模擬「看手臂動起來」的驗證
```

## 技術選型

- **編排大腦**：Claude `claude-sonnet-4-6`（Anthropic SDK，含 retry）
- **VLA policy**：官方微調 checkpoint `HuggingFaceVLA/smolvla_libero`（不自己微調）
- **模擬**：Week 1 用 Meta-World（Mac 本機，看手臂動）；Week 2 用 LIBERO（Kaggle，真正
  用 SmolVLA 驅動）。LIBERO 僅 Linux，故 Mac 本機只能跑 mock 版。
- **執行管線**：`libero_skills.execute()` 以 subprocess 呼叫官方 `lerobot-eval`，重用整條
  已驗證管線換取穩定性（代價是每次重載 policy、較慢）。

## 誠實的限制

- **慢**：每次 execute 經 subprocess 重載 policy（~3 min/task）。可改為「載 policy 一次 +
  直接呼叫 `eval_policy()`」加速。
- **成功率受限於官方 checkpoint，未自行微調**：`libero_object` 單物件抓放 100%；`libero_10`
  長程任務官方 checkpoint 僅 ~20%（夠當「真實失敗」素材、不適合 happy-path）。未涵蓋雙臂/堆疊。
- **Week 1 是 metaworld expert 腳本、非 SmolVLA**：SmolVLA 的 SO101 action/obs space 與
  metaworld 4-dim 不相容，zero-shot 接不上；真正用 SmolVLA 驅動是在 Week 2 的 LIBERO。
- **sim-to-real gap**：全程純模擬，未上實體手臂。
