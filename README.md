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

## agent 行為證據：①〜⑤ 真 LIBERO rollout，⑥ plan-only 拆解（零 GPU）

| 情境 | 指令 | agent 行為 | 證明的能力 |
|---|---|---|---|
| 成功 | 把字母湯罐頭收進籃子 | `execute(0)` → ✅ | 語意對應 + 真閉環 |
| 多步 | 字母湯和番茄醬**都**收進籃子 | `execute(0)`→`execute(5)` → ✅ | 任務分解 + 排序 |
| 拒絕 | 幫我把桌子擦乾淨 | ⛔ 超出範圍（不跑 rollout） | 能力邊界自覺 |
| 澄清 | 把**那個東西**收起來 | ❓ 反問 + 列選項 | 不幻覺、會反問 |
| 真實失敗 | （`libero_10`）兩樣都收進籃子 | 真 rollout 失敗×3 → 🛑 誠實 abort | 失敗當一等公民（真證據） |

前四種在 `libero_object`；第五種在 `libero_10` 長程任務（官方 checkpoint 實測僅 ~5% 成功率，
失敗訊號來自真 ground-truth、非 mock `--fail-first`）。跑了 4 輪真 rollout 一致 abort——救回邏輯
由確定性單元測試保證，不靠僥倖拍影片。

**⑥ plan-only 拆解驗證（零 GPU，本機秒回）**：把「拆解人話→任務序列」這層單獨拎出來、**不跑
rollout**，專測更難的語言推理——語意分群（「把**所有醬料**都收」→ agent 自動挑出醬料類物件）、
排除否定（「**除了**番茄醬…」）、3+ 物件、排序約束。真 Claude log 還揭露兩個自然語言坑：「番茄醬」
是**真歧義**（繁中口語偏指 ketchup，但選單同時有 ketchup 與 tomato sauce），且「醬料」的**類別邊界
會浮動**（cream cheese 在不同次跑之間算不算醬料會翻覆）——正是該觸發澄清、policy 碰不到的問題。
`python demo_plan.py "<指令>"` 本機即可跑，`--live` 可在 Kaggle 對照真選單；細節見 demo-results.md 情境⑥。

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
.venv/bin/python -m pytest          # 57 tests，全程 mock，零 API 成本
```

### 量測 rollout 加速（Kaggle）

```bash
python bench_rollout.py                 # in-process vs subprocess 牆鐘時間 + 成敗 parity
python bench_rollout.py --policy groot  # 切 GR00T N1.5（需 Ampere+ GPU，見 spec 附錄 A）
```

完整 Kaggle step-by-step 見 [`docs/kaggle-bench-rollout-walkthrough.md`](docs/kaggle-bench-rollout-walkthrough.md)。

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
  plan_only.py      軸 A：plan-only 拆解驗證（decompose_only/format_plan，純函式、零 lerobot）
  rollout_engine.py L1/L2 加速接縫：RolloutEngine（InProcess 常駐／Subprocess baseline／GR00T builder）
demo.py             本機 mock CLI
demo_libero.py      Kaggle 真 LIBERO CLI
demo_plan.py        plan-only CLI：只驗拆解、不跑 rollout（本機零 GPU）
bench_rollout.py    Kaggle 速度實測 + 成敗 parity（in-process vs subprocess；可 --policy groot）
tests/              57 tests（TDD，全程 mock）
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

- **慢（已解）**：subprocess 版每次 execute 重載 policy（~270s/task）。已實作
  `InProcessRolloutEngine`（policy/env 常駐複用、直接呼叫 `eval_policy()`）——Kaggle T4 實測
  穩態約 **2.0× 加速**、每 task 省 ~139s，成敗 parity 全 ✅（見 demo-results.md ⑦）。
- **成功率受限於官方 checkpoint，未自行微調**：`libero_object` 單物件抓放 100%；`libero_10`
  長程任務官方 checkpoint 實測僅 ~5%（夠當「真實失敗」素材、不適合 happy-path）。未涵蓋雙臂/堆疊。
- **Week 1 是 metaworld expert 腳本、非 SmolVLA**：SmolVLA 的 SO101 action/obs space 與
  metaworld 4-dim 不相容，zero-shot 接不上；真正用 SmolVLA 驅動是在 Week 2 的 LIBERO。
- **sim-to-real gap**：編排層（L3）全程純模擬，未上實體手臂的 closed-loop。但已另闢一條
  **real-robot fine-tune 軌跡**：用社群真 SO-101 遙操作資料集（`lerobot/svla_so101_pickplace`，
  50 episodes）在 Kaggle 把 `smolvla_base` fine-tune 跑通、loss 收斂、checkpoint 上 HF Hub
  （[`Kaminoikari/smolvla-so101-pickplace-ft`](https://huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft)）。
  這證明了「讀真資料→微調 VLA→出 checkpoint」整條 pipeline；**仍缺**的是實體手臂的
  closed-loop 成功率（要接真 SO-101 或對應 sim 才量得到）。流程見
  [`docs/kaggle-finetune-smolvla-so101.md`](docs/kaggle-finetune-smolvla-so101.md)。
