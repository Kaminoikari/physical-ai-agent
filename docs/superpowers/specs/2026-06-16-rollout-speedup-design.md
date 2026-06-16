# 自我迭代閉環加速設計（rollout speedup）

> 日期：2026-06-16
> 目標：在不破壞「真 ground-truth 成敗」可信度的前提下，把自我迭代閉環的速度提升一個數量級，
> 並維持成本最低（pretrained 凍結權重、不做 GPU 訓練）。

## 1. 問題

現況 `LiberoSkillInterface.execute()` 每跑一個 task 都用 `subprocess.run(["lerobot-eval", ...])`，
**每次都從硬碟重載 SmolVLA checkpoint + 重啟 LIBERO env（~3 min/task）**。自我迭代閉環會把
「規劃 → 跑 → 觀察 → 再規劃」重複很多輪，這個冷啟動成本被乘起來，是慢的根因——慢的不是
GPU、不是動作本身，是反覆的重載。

## 2. 範圍

三項合一份 spec（皆圍繞「迴圈速度」）：

- **#1 policy/env 常駐**：subprocess 重載 → 載一次、常駐複用。（Kaggle-only 才能實測）
- **#2 plan-only 快迴圈 + 完成記憶**：思考迭代不跑 rollout；同一 run 內已完成子任務不重跑。（本機可測）
- **#3 rollout 微優化**：不寫 mp4、並行 batch、多回合——大多靠設定即得，不手刻。（Kaggle-only）
- **#4 GR00T N1.5 swap 接縫（前沿 opt-in）**：builder 參數化成 `(policy_type, policy_path)`，
  讓同一引擎能在 SmolVLA（便宜預設）與 GR00T N1.5（前沿）間切換 + 一次實測。（Ampere+ GPU 才實測）

**非範圍**：閉環 replan 的「再觀察後重新拆解」語意升級、affordance grounding、跨 episode 的
self-improvement memory、policy 微調訓練——皆為後續獨立 spec。本 spec 只到「能一鍵換 GR00T 並
跑通一次」，不含在 GR00T 上做完整自我改進迭代。

## 3. 選型決策（為什麼走 C）

| 做法 | 內容 | 取捨 |
|---|---|---|
| A | 直接用 lerobot `eval_policy(env, policy, …)` + 常駐 holder | 低風險、重用官方已驗證 eval；控制權較少 |
| B | 自己手刻 rollout step loop | 速度天花板高；**重寫官方 eval = 重寫「真成敗」信譽地基**，風險高、維護重 |
| **C（採用）** | A 打底，#3 只補「測得出瓶頸」的那幾刀，且包在 flag 後 | 鎖定 #1 數量級大勝、免費收 #3 大半、把唯一不確定性隔離成有退路的小區塊 |

**C 的保證範圍**：C 是「成本 × 效能 × 可信度」目標函數下的風險調整最強解。它壓在一個假設上——
`eval_policy` 能用常駐 env+policy 反覆呼叫；此假設已由官方原始碼確認（見 §4）。萬一某版行為不符，
退路是只把 `InProcessRolloutEngine.run` 內那一小段換成薄手刻 step loop，**接縫與成敗判定不動**。

## 4. 已驗證的 lerobot API 事實（grounding）

來源：`huggingface/lerobot` `src/lerobot/scripts/lerobot_eval.py`（main，2026-06 查證）。

- `eval_policy(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor,
  n_episodes, max_episodes_rendered=0, videos_dir=None, return_episode_data=False, start_seed=None)`
  **吃預建好的 env + policy**（常駐複用成立）。
- 回傳 `dict`：`"per_episode"`（每集 `success` 等）、`"aggregated"`（含 **`pc_success`**）、可選 `"video_paths"`。
- **影片預設關閉**：`max_episodes_rendered=0` 且 `videos_dir=None` → 不渲染、不寫檔。快迴圈天然快。
- `eval_policy_all(...)` 的聚合鍵是 `"overall"`（含 `pc_success`），即現行 `_run_eval` 讀的欄位。
- `rollout(env, policy, …)` 直接回傳 `"success"`/`"done"` tensor——B 退路所需的低階接口存在。
- **重要細節**：新版 `eval_policy` 需四條 processor pipeline（`env_pre/post`、`pre/post`），與 policy 一起建。
  故常駐 holder 要握 `{policy + 四條 processor + 各 task 的 env}`，非只 policy。

來源：`huggingface/lerobot` GR00T 整合（HF blog `nvidia/nvidia-isaac-gr00t-in-lerobot`，2026-06 查證）。

- **GR00T N1.5 是標準 lerobot policy**：`policy.type=groot`、checkpoint `nvidia/GR00T-N1.5-3B`，
  與 SmolVLA、pi0、pi0.5 走**同一 policy 介面**，`lerobot-eval --policy.path=` 用法一致 → 換 policy
  只是換 builder 參數，`InProcessRolloutEngine` 與 `eval_policy` 呼叫**不改**。
- **依賴**：`pip install -e ".[libero,groot,dev,test]"` + flash-attention + torch/torchvision；支援 LIBERO。
- **硬體現實（誠實警告）**：GR00T N1.5 = **3B 參數**（SmolVLA 450M），官方在 H100 / A6000 測；flash-attn
  需 Ampere+。**Kaggle T4（Turing, sm_75）很可能跑不動或爆 VRAM** → GR00T 實測需租 A6000/L4 等 Ampere+ GPU。
  故 **SmolVLA 為便宜預設、GR00T 為前沿 opt-in**，非取代關係。

## 5. 設計

### 5.1 RolloutEngine 接縫（#1 / #3）

在 `LiberoSkillInterface` 與「怎麼跑 rollout」之間切一條可注入接縫：

```
RolloutOutcome: { success: bool, pc_success: float, video_path: str | None }

RolloutEngine（協定）
  run(task_id: int, *, save_video: bool, n_episodes: int) -> RolloutOutcome

├ SubprocessRolloutEngine   保留現行 lerobot-eval subprocess（Kaggle 速度 baseline）
├ InProcessRolloutEngine     #1 主角：lazy 建構 + 常駐複用，內呼 eval_policy
└ FakeRolloutEngine          本機 mock，腳本化 outcome，不碰 lerobot
```

`InProcessRolloutEngine`：

- 持有一個可注入的 **builder**（負責 `make_policy` + 四條 processor + `make_env`）；首次 `run` 才呼叫，
  之後快取。policy/processor 全 session 建一次；env 以 `dict[task_id, VectorEnv]` 快取、首次用到該 task 才建。
- **builder 參數化成 `(policy_type, policy_path)`（#4 GR00T 接縫）**：預設
  `("smolvla", "HuggingFaceVLA/smolvla_libero")`；傳 `("groot", "nvidia/GR00T-N1.5-3B")` 即切到 GR00T N1.5。
  因 lerobot 統一 policy 介面，**不需 GR00T 專屬類別**，`run`/`eval_policy` 一行不改——這是最高槓桿的接法。
- `run` 呼叫 `eval_policy(env, policy, …processors, n_episodes,
  max_episodes_rendered=(N if save_video else 0), videos_dir=…)`，回傳 `aggregated.pc_success` 包成 `RolloutOutcome`。
- builder 與快取邏輯分離：builder 是碰 lerobot 的薄膠水（Kaggle-only），快取/旗標路由是純邏輯（本機可測）。

`LiberoSkillInterface`：

- `__init__` 多收 `engine: RolloutEngine | None`（預設 `InProcessRolloutEngine`）。
- `execute` 委派 `engine.run(...)`；**`success_threshold` 判定保留在 interface**（`pc_success ≥ threshold → ok`）——
  成敗語意一行不改。
- 對 `agent.py`（L3）完全透明。

### 5.2 plan-only 快迴圈 + 完成記憶（#2）

- **plan-only 快迴圈**：重用 `plan_only.py`。自我迭代時規劃/重規劃走 `decompose`（+ 可選計畫自評），
  **不跑 rollout**；計畫穩定後才進真 rollout 驗收。做成 `Agent` 的模式旗標或薄包裝，不污染主迴圈。
- **完成記憶（completed-task memo）**：`Agent.run` 內維護 `completed: set[str]`；`execute(task)` 成功後標記，
  replan 若再排入同一子任務則跳過並記 log。**確定性安全**（同一 episode 內已放好的物件維持放好），取代
  「隨機 rollout 結果快取」（後者語意不誠實，已棄）。

### 5.3 改動清單

- `agent/libero_skills.py`：抽出 `RolloutEngine` 協定 + 三實作；`LiberoSkillInterface` 改委派。
- `agent/agent.py`：`run()` 加 completed memo 與 plan-only 模式旗標。
- `tests/`：新增 RolloutEngine / memo / plan-only 模式的 mock 測。
- Kaggle 腳本：`bench_rollout.py`（速度實測 + 成敗 parity；可選 `--policy groot` 跑 GR00T N1.5 一次）。

## 6. 測試策略

**本機（TDD，擴充現有 42 測，零 GPU/API）：**

- `FakeRolloutEngine` → 測 `execute` 套 `success_threshold` 正確、`save_video` 旗標下傳。
- fake builder → 測「跑 N 次 builder 只呼叫 1 次」「同 task_id env 第二次命中快取」「builder 收到
  正確的 `(policy_type, policy_path)`，預設 SmolVLA、可換 GR00T」。
- `FakeSkills` → 測完成記憶（已成功子任務 replan 被跳過、`execute` 只呼叫一次）。
- plan-only 模式 → 斷言 `decompose` 跑、`execute` 完全沒被呼叫。

**Kaggle（一次性腳本 `bench_rollout.py`，非 pytest）：**

- **速度實測**：同批 task 跑 `SubprocessRolloutEngine` vs `InProcessRolloutEngine`，印每 task + 總牆鐘時間。
- **成敗 parity**：同幾個 task 斷言兩引擎給出**相同成敗判定** → 鐵證加速沒動搖 ground-truth 語意。

**四個 best practice**：①建構與邏輯分離（本機可證邏輯最大化）②TDD red→green ③Kaggle parity test（加速不改壞結果的保證書）④只在 RolloutEngine 接縫 mock、不深入 mock lerobot。

## 7. 風險與退路

| 風險 | 退路 |
|---|---|
| `eval_policy` 某版不能常駐複用 | 僅 `InProcessRolloutEngine.run` 內改薄手刻 step loop（用 `rollout()`），接縫與判定不動 |
| 常駐 env 記憶體吃緊 | env 按 task_id lazy 建 + 可設上限驅逐；libero_object 僅 10 task，實務無虞 |
| in-process 與官方 eval 語意飄移 | Kaggle parity test 擋下；不符即回退 subprocess baseline |
| 完成記憶誤跳尚未真正完成的任務 | memo 僅在 rollout 回報 success 後寫入；assume_success 模式不寫 memo |
| GR00T N1.5 在 Kaggle T4 跑不動（Turing 無 flash-attn / VRAM 不足） | 接縫與本機測不受影響（builder 參數化已驗）；GR00T 實測延到租 A6000/L4，SmolVLA 維持便宜預設 |

## 8. 成功判準

- 本機新增測全綠，既有 42 測不破。
- Kaggle 實測：in-process 每 task 牆鐘時間 ≪ subprocess（目標數量級）。
- Kaggle parity：兩引擎成敗判定一致。
- demo-results 補上「加速前/後」實測數字。
- #4：本機測證明 builder 可參數化切 GR00T；Ampere+ GPU 上 `bench_rollout.py --policy groot` 至少跑通一次
  （或明確記錄「T4 不支援、待租 GPU」的誠實結果）。
