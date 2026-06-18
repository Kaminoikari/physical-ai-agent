# 在 Kaggle fine-tune SmolVLA（real SO-101 資料集，零硬體）

> 目標：不買任何手臂，在 Kaggle GPU 上把 **real-robot imitation-learning pipeline 的核心**
> 親手跑一遍——拿官方 `lerobot/smolvla_base`（450M）在社群採的**真** SO-101 pick-place
> 資料集上 fine-tune，產出自己的 checkpoint，並做離線評估。指令以 lerobot 官方
> `docs/source/smolvla.mdx`（0.5.x）為準。
>
> **誠實的範圍界定（先讀）：**
> - ✅ **能拿到**：data loading → fine-tune → 產出 checkpoint → 離線評估這條鏈，全程零硬體。
>   這就是「A 軸：pipeline 技能」，可寫進作品集、面試直接講。
> - ⚠️ **拿不到**：closed-loop 真實成功率。官方 eval（`lerobot-record`）要接**實體** SO-101
>   （`/dev/ttyACM0`），沒手臂就沒有真 rollout 成功率。本文 eval 段用「loss 曲線 + 離線
>   action 誤差」當誠實替代——**這正是之後要不要買 SO-101 的真正分界（B 軸：真硬體手感）**。

## 0. Notebook 設定（網頁操作）
1. kaggle.com → Create → **New Notebook**。
2. 右側 **Settings**：
   - **Accelerator** → **GPU T4 x2**（或 P100）。Kaggle 沒有 A100；官方建議的
     `batch_size=64 / 20k steps / ~4hr on A100` 在 T4 上要縮小（見步驟 4）。
   - **Internet** → **On**（要 pip install、拉 base model 與資料集）。
3. 每週免費約 30 GPU 小時。本文「冒煙測試」設定約 20–40 分鐘，足夠驗證 pipeline 跑通。

## 1. 裝 lerobot + smolvla extras（第一個 cell）
```python
!pip -q install "lerobot[smolvla]"
import torch, lerobot
print("lerobot", lerobot.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
> 若 base model 載入報設定不相容，改裝 main：
> `!pip -q install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@main"`

## 2. 登入 HF Hub（第二個 cell）
拉 `smolvla_base`、之後把自己的 checkpoint push 回 Hub 都需要。把 HF token 放進
**Kaggle Secrets**（Add-ons → Secrets → 新增 `HF_TOKEN` 並 Attach），不要明文寫進 notebook。
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
HF_USER = "你的_hf_帳號"   # ← 改成自己的，push checkpoint 會用到
print("HF login OK")
```

## 3. 看一眼資料集（第三個 cell，sanity check）
`lerobot/svla_so101_pickplace`：50 episodes、11,939 frames、雙相機
（`observation.images.up`／`observation.images.side`）、6-dim 關節 state。先確認讀得到、
看清 observation/action 結構，再開訓練。
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/svla_so101_pickplace")
print("episodes:", ds.num_episodes, "| frames:", ds.num_frames)
print("features:", list(ds.features.keys()))
print("一個 sample 的 keys:", list(ds[0].keys()))
```
> 這步本身就是 pipeline 的一環：**搞懂資料長相**（相機數、state/action 維度）才知道
> fine-tune 接不接得上。SmolVLA 會自動對映多相機 + state + 語言指令。

## 4. Fine-tune（第四個 cell）——先冒煙測試，再決定要不要拉長
T4 16GB 跑不動官方的 `batch_size=64`。**第一輪先用小設定驗證「能跑、loss 會降」**：
先一個獨立 cell 清殘留（**不要**和下面的訓練指令塞同一個 cell——兩個 `!` 加空行會讓
Jupyter 續行解析錯亂報 `IndentationError`）：
```python
!rm -rf /kaggle/working/smolvla_so101   # 清掉上次失敗殘留，免得報「已存在」
```
再另一個 cell 跑訓練：
```python
!lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --batch_size=8 \
  --steps=2000 \
  --output_dir=/kaggle/working/smolvla_so101 \
  --job_name=smolvla_so101_smoke \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```
> - **`--rename_map` 不可省**：`smolvla_base` 預設期待 3 個相機 `camera1/2/3`，本資料集只有
>   `up`／`side` 兩個。lerobot 的驗證規則是「資料集相機 ⊆ policy 期待相機」才放行，所以把
>   `up→camera1`、`side→camera2`（JSON 用單引號包住），湊成子集即通過，第三個相機 smolvla
>   自動留空。不加會報 `Feature mismatch ... Missing: camera1/2/3, Extra: side/up`。
> - **`--policy.push_to_hub=false` 不可省**：lerobot-train 預設會想把模型 push 到 Hub，
>   沒給就會報 `'policy.repo_id' argument missing`。冒煙階段關掉它，checkpoint 留在
>   `output_dir`，要留再用步驟 6 手動 push。
> - **冒煙通過後**（loss 明顯下降、無 OOM），要練「真的學起來」再把 `--steps` 拉到
>   `10000`～`20000`、`--batch_size` 在不 OOM 前提下調大（T4 大概到 8～16）。20k steps 在 T4
>   會遠比 A100 的 4hr 久，分次跑或用 checkpoint 續訓。
> - OOM 就把 `--batch_size` 降到 4，或加梯度累積相關選項（`lerobot-train --help` 查）。
> - 想看訓練曲線就把 `--wandb.enable=true` 並先 `wandb login`（Secrets 放 `WANDB_API_KEY`）。

## 5. 離線評估（第五個 cell）——沒手臂時的誠實替代
官方 closed-loop eval 要接實體手臂。沒手臂時，**最誠實的「有沒有學起來」訊號**有兩個：
1. **訓練 loss 曲線**：有沒有穩定下降（wandb 或 train log）。這是最低標。
2. **離線 action 誤差**：拿訓練時沒看過的 frame，比對「policy 預測的 action」vs「資料集
   裡的真 action」。誤差小 = 模型在模仿。下面是可改的草稿（依你的 lerobot 版本微調
   API 名稱）：
> **checkpoint 路徑**：lerobot 存在 `output_dir/checkpoints/last/pretrained_model/`，
> 不是 output_dir 根目錄。先 `!ls -R /kaggle/working/smolvla_so101/checkpoints/last` 確認。
```python
# 草稿：離線 action 誤差（held-out 概念示意，跑前對照你版本的 policy/dataset API）
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "/kaggle/working/smolvla_so101/checkpoints/last/pretrained_model"
policy = SmolVLAPolicy.from_pretrained(CKPT)
policy.eval().to("cuda")

ds = LeRobotDataset("lerobot/svla_so101_pickplace")
errs = []
for i in range(0, 200, 20):              # 抽樣幾個 frame
    batch = {k: (v.unsqueeze(0).to("cuda") if torch.is_tensor(v) else v)
             for k, v in ds[i].items()}
    with torch.no_grad():
        pred = policy.select_action(batch)        # 預測 action
    gt = ds[i]["action"].to("cuda")
    errs.append(torch.nn.functional.mse_loss(pred.squeeze(), gt).item())
print("離線 action MSE（抽樣）:", sum(errs)/len(errs))
```
> 這是 offline metric，**不等於**真實任務成功率——真成功率只有接上手臂或對應 sim 才量得到。
> 文件這樣寫，是為了不浮誇：你練到的是 pipeline，不是「驗證了 real-world 成功」。

## 6. 保存 checkpoint（第六個 cell）——Kaggle 不持久
Kaggle session 重置會清掉 `/kaggle/working`。要留下訓練成果，push 到自己的 HF Hub：
```python
from huggingface_hub import HfApi
api = HfApi()
repo = f"{HF_USER}/smolvla-so101-pickplace-ft"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="/kaggle/working/smolvla_so101/checkpoints/last/pretrained_model",
    repo_id=repo, repo_type="model")
print("pushed:", repo)
```
> 之後若真買了 SO-101，就能直接 `--policy.path=你的帳號/smolvla-so101-pickplace-ft`
> 接 `lerobot-record` 做真 closed-loop eval——那一刻才補上 B 軸。

## 已知雷與對策
- **沒開 Internet** → pip / 拉 model / 拉 dataset 全失敗。先確認 Settings。
- **CUDA OOM** → `--batch_size` 降到 4；SmolVLA 450M + T4 16GB 要小 batch。
- **base model 載入設定不相容** → 改裝 git main（見步驟 1 註解）。
- **離線 eval 腳本 API 對不上** → 不同 lerobot 版本 `select_action` / policy 類名會變，
  以 `lerobot-train --help` 與安裝後的 `lerobot.policies` 實際模組為準（這段是草稿，非保證可跑）。
- **Session 逾時** → 訓練未完先 push 已存的 checkpoint；長訓練分段續訓。
- **20k 步在 T4 跑不完** → 實測 T4 約 2.34 s/step，2000 步 ~78 分鐘，20k 步 ~13 小時，
  超過 Kaggle 單 session 12 小時上限。要練到 paper 等級得：加 `--save_freq` 跨 session
  `--resume=true` 續訓，或租 RunPod A6000/A100（見 rollout-speedup spec 附錄 A）。冒煙 2000
  步足以證明 pipeline 跑通、loss 收斂。

## 你練到了什麼 / 還缺什麼（對照你的目標）
- ✅ **A 軸 pipeline 技能（完整拿到）**：讀真 robot 資料集 → 在預訓練 VLA 上 fine-tune →
  產出/上傳 checkpoint → 離線評估。這條鏈跑通，你就**真的會** real-robot 學習的核心流程。
- ⚠️ **B 軸真硬體手感（仍缺）**：遙操作親手採集、真 closed-loop 成功率、校正/latency/gripper
  滑動等真實 failure mode——只有接上實體 SO-101（或對應 sim）才拿得到。
- 👉 **決策點**：做完這份指南，你已用近零成本拿到八成價值。剩下兩成（B 軸）值不值得花
  ~$250 買 SO-101，取決於你是否要把 real-robot pipeline 變成「可受僱的技能」而不只是 side
  project。建議先做完這份、再評估，那時買就是有明確產出的投資、不是賭。
