# Week 2 — Kaggle 上跑真 LIBERO（用官方 smolvla_libero checkpoint）

> 目標（de-risk 里程碑）：在 Kaggle 免費 T4 上，用官方微調好的 `HuggingFaceVLA/smolvla_libero`
> 跑 `lerobot-eval`，在 LIBERO 模擬裡看到**真的成功的 pick-place**。先不自己微調。
> 指令皆以 lerobot 官方 `docs/source/libero.mdx`（0.5.x）為準。

## 0. 建立 Notebook 與設定（網頁操作）
1. kaggle.com → Create → **New Notebook**。
2. 右側 **Settings**：
   - **Accelerator** → **GPU T4 x2**（或 P100）。
   - **Internet** → **On**（不開無法 pip install / 下載 checkpoint）。
3. 每週免費額度約 30 GPU 小時；eval 很省，這步花不到 1 小時。

## 1. 裝系統 EGL 庫 + lerobot（第一個 cell）
```python
# LIBERO 用 MuJoCo/robosuite，headless 渲染需要 EGL 系統庫
!apt-get -qq update && apt-get -qq install -y \
    libegl1 libgl1-mesa-glx libosmesa6 libglfw3 libglew-dev patchelf > /dev/null
# 官方 release（與 docs 對應）；含 smolvla 與 libero extras
!pip -q install "lerobot[smolvla,libero]"
```
> 若 checkpoint 載入時報設定不相容，改裝 main：
> `!pip -q install "lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git@main"`

## 2. 設渲染後端 + 健檢（第二個 cell）
```python
import os
os.environ["MUJOCO_GL"] = "egl"   # 官方指定的 headless 後端
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch, lerobot
print("lerobot", lerobot.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
# 確認 libero env 可建立（會觸發 LIBERO 資產下載，第一次較久）
from lerobot.envs.configs import LiberoEnv
print("LiberoEnv OK, default task:", LiberoEnv().task)
```

## 3. 跑 eval：官方 checkpoint 在 LIBERO（第三個 cell）
先用最小設定快速看到結果（單一 task、1 episode）：
```python
!MUJOCO_GL=egl lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --env.max_parallel_tasks=1 \
  --output_dir=/kaggle/working/eval_libero
```
> `--env.task_ids="[0]"` 只跑該 suite 第一個任務，最快。要完整 benchmark 再拿掉、把
> `--env.task` 換成四 suite 逗號清單、`--eval.n_episodes=10`（會很久）。
> 若 checkpoint 的 action 參數化不符，加 `--env.control_mode=absolute` 試試（預設 relative）。

## 4. 看影片（第四個 cell）
```python
import glob
from IPython.display import Video
vids = sorted(glob.glob("/kaggle/working/eval_libero/**/*.mp4", recursive=True))
print(vids)
Video(vids[0], embed=True, width=480) if vids else print("沒找到影片，檢查 eval log")
```

## 成功訊號
在影片裡看到 LIBERO 的機械臂**抓起物件並放到目標**（成功率看 eval 印出的 success rate）。
達到即代表「真 L1（LIBERO + 官方 smolvla）在雲端跑通」，可進到把它包成 `LiberoSkillInterface`。

## 已知雷與對策
- **沒開 Internet** → pip / 下載全失敗。先確認 Settings。
- **EGL 報錯（libEGL / GLEW / GLFW）** → 確認 cell 1 的 apt 套件都裝了、`MUJOCO_GL=egl` 有設；
  必要時試 `os.environ["MUJOCO_GL"]="osmesa"`（較慢但純 CPU 渲染、相容性高）。
- **LIBERO 第一次跑很久** → 在下載 BDDL/資產，正常。
- **checkpoint 載入設定錯誤** → 改裝 git main（見步驟 1 註解）。
- **Session 逾時/重置** → Kaggle 不持久；eval 結果存 `/kaggle/working`（可下載）。要保留長期產物就
  push 到自己的 HF Hub（`huggingface_hub` login + `--output_dir` 後手動上傳，或之後微調時用 `--policy.repo_id`）。

## 下一步（接回你的 agent 層）
eval 通過後，新增 `agent/libero_skills.py`：實作與 `agent/skills.py` 同簽章的
`pick/place/query`，內部驅動 LIBERO env（pick/place 對映到 policy rollout 片段，query 讀
LIBERO 的物件座標真值）。L3（`agent.py`）完全不動，即把今天的編排層接上真 sim。
