# 在 Kaggle 驗 rollout 加速 + 成敗 parity（PR #1 把關）

> 目標：在 Kaggle GPU 上實測 `bench_rollout.py`，證明兩件事再 merge PR #1：
> ① **parity**——新的 in-process 引擎跑出的成功/失敗，與舊的 subprocess baseline **一致**
> （加速沒改壞正確性）；② **speed**——in-process 比 subprocess **快多少**（牆鐘時間）。
> 所有單元測試都是 mock，這支是唯一在真 GPU 上驗證重構的環節。

## 為什麼需要這步
PR #1 把 LIBERO 執行從「每個 task 用 subprocess 重載 policy（~3 min/task）」改成
「policy 載一次、常駐重用」。57 個測試全綠但**全是 mock**，沒有任何一個真的在 GPU 上跑過
in-process 引擎。merge 前必須用真環境確認：**結果一致（parity）+ 真的變快（speed）**。

## 0. Notebook 設定（同 week2-kaggle-libero-setup）
1. kaggle.com → New Notebook。
2. 右側 **Settings**：**Accelerator → GPU T4 x2**、**Internet → On**。
3. 若這個 session 已照 `week2-kaggle-libero-setup.md` 裝好 lerobot+EGL、寫好
   `~/.libero/config.yaml`、跑過一次 eval（assets 已快取），可**跳到步驟 3**。

## 1. 裝系統 EGL 庫 + lerobot（第一個 cell）
```python
!apt-get -qq update && apt-get -qq install -y \
    libegl1 libgl1-mesa-glx libosmesa6 libglfw3 libglew-dev patchelf > /dev/null
!pip -q install "lerobot[smolvla,libero]"
```

## 2. 渲染後端 + LIBERO 設定（第二個 cell）
```python
import os, yaml
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

broot = "/usr/local/lib/python3.12/dist-packages/libero/libero"
cfg = {"benchmark_root": broot,
       "bddl_files": os.path.join(broot, "bddl_files"),
       "init_states": os.path.join(broot, "init_files"),
       "datasets": os.path.join(os.path.dirname(broot), "datasets"),
       "assets": os.path.join(broot, "assets")}
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
yaml.safe_dump(cfg, open(os.path.expanduser("~/.libero/config.yaml"), "w"))
print("libero config ready")
```

## 3. clone repo 到 **PR #1 的 branch**（第三個 cell）
`bench_rollout.py`、`agent/rollout_engine.py` 還在 `feature/rollout-speedup`，尚未 merge 到
main，所以**要 clone 這條 branch**。需要 GitHub PAT（放 Kaggle Secrets，見
week2-kaggle-agent-libero-run.md 步驟 1）。
```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
_pat = UserSecretsClient().get_secret("GITHUB_PAT")
subprocess.run(
    ["git", "clone", "--branch", "feature/rollout-speedup",
     f"https://{_pat}@github.com/Kaminoikari/physical-ai-agent.git"],
    check=True, capture_output=True,
)
print("cloned feature/rollout-speedup")
```
> bench 本身不呼叫 Claude，不需要 `ANTHROPIC_API_KEY`。

## 4. 跑 bench（第四個 cell）
```python
import os
os.environ["MUJOCO_GL"] = "egl"
%cd /kaggle/working/physical-ai-agent
!python bench_rollout.py --tasks 0,1,2
```
參數（預設值見括號）：
- `--tasks 0,1,2`：跑哪幾個 task id（逗號分隔）。task 越多，in-process 的「載一次」攤提越明顯。
- `--suite libero_object`：哪個 suite。
- `--threshold 50.0`：pc_success ≥ 此值算成功（parity 以此二值化比對）。
- `--policy smolvla`：baseline 對比僅 smolvla 有（groot 無 subprocess baseline）。

## 5. 怎麼讀輸出
```
== in-process [smolvla] ==
  task 0: pc_success=100.0  82.3s      ← 第一個 task 含一次性載 policy
  task 1: pc_success=100.0  11.4s      ← 之後重用，快很多
  task 2: pc_success=0.0    11.0s

== subprocess [baseline] ==
  task 0: pc_success=100.0  176.5s     ← 每個 task 都重載 policy
  task 1: pc_success=100.0  175.9s
  task 2: pc_success=0.0    176.2s

== 成敗 parity（門檻 50）==
  task 0: in=True  sub=True  ✅
  task 1: in=True  sub=True  ✅
  task 2: in=False sub=False ✅
```
（**以上數字為示意格式，非實測**；實際數字以你的 run 為準，填回下方文件。）

**判讀**：
- **parity 全 ✅** = in-process 與 subprocess 對每個 task 的成敗判定**完全一致** → 加速沒改壞正確性。**這是 merge 的硬條件**。
- **speed**：in-process 第一個 task 慢（含一次性載 policy），第二個起明顯變快；subprocess 每個 task 都重載故每個都慢。task 數越多，總時間差距越大——這就是加速的證據。
- 若**出現任何 ❌**（成敗不一致）→ **先別 merge**，把完整輸出貼回來排查（多半是 in-process 的 env/processor 接線與 subprocess 預設不同所致）。

## 6. 把數字填回文件（merge 前最後一步）
1. `docs/demo-results.md` 情境⑦：填入實測的 in-process vs subprocess 時間表與 parity 結果。
2. `README.md` 誠實的限制「慢」那條：把「可改為…加速」更新為實測加速倍數。
3. commit（`docs:` 開頭）後，PR #1 即可 merge。

## 已知雷
- **沒開 Internet / 沒裝 EGL** → 同 week2，先回步驟 0–1。
- **clone 失敗** → 確認 `GITHUB_PAT` secret 已 attach、且對 repo 有 Contents:Read。
- **第一個 task 特別久** → 正常（含載 policy + 第一次建 env + 下載 checkpoint）。
- **task 2 成功率 0** → 不一定是錯；`libero_object` 個別 task/episode 本就可能失敗，重點是
  **兩引擎一致**，不是「都要成功」。
- **groot baseline 沒印** → 預期行為：groot 無 subprocess 對照，只測 in-process（且需 Ampere+ GPU，T4 跑不動，見 rollout-speedup spec 附錄 A）。
