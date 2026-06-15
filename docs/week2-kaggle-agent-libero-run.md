# 在 Kaggle 跑 agent × 真 LIBERO 端到端 demo

> 前置：同一個 Kaggle notebook 已裝好 `lerobot[smolvla,libero]`、EGL 系統庫、寫好
> `~/.libero/config.yaml`、且跑過一次 eval（assets 已快取）。若是新 session，請先重跑
> 那些步驟（見 week2-kaggle-libero-setup.md）。

## 1. 準備兩個密鑰（一次性）
1. **GitHub PAT**：github.com → Settings → Developer settings → Fine-grained tokens →
   新增一個，只給 `physical-ai-agent` 這個 repo 的 **Contents: Read-only**。複製 token。
2. **Kaggle Secrets**：notebook 右側 Add-ons → **Secrets** → 新增兩個並 **Attach 給本 notebook**：
   - `ANTHROPIC_API_KEY` = 你的 Anthropic key
   - `GITHUB_PAT` = 上面的 GitHub token

## 2. 載入密鑰（cell）
```python
import os
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
os.environ["ANTHROPIC_API_KEY"] = sec.get_secret("ANTHROPIC_API_KEY")
_pat = sec.get_secret("GITHUB_PAT")
print("secrets loaded")
```

## 3. clone private repo + 裝 agent 相依（cell，用 subprocess 避免 token 外洩到輸出）
```python
import subprocess
subprocess.run(
    ["git", "clone", f"https://{_pat}@github.com/Kaminoikari/physical-ai-agent.git"],
    check=True, capture_output=True,
)
print("cloned")
!pip -q install anthropic python-dotenv
```

## 4. 確保 LIBERO 設定存在（新 session 才需要；同 session 可略）
```python
import os, yaml
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

## 5. 跑端到端 demo（cell）
```python
import os
os.environ["MUJOCO_GL"] = "egl"
%cd /kaggle/working/physical-ai-agent
!python demo_libero.py "把字母湯罐頭收進籃子"
```
> 第一次會列出 libero_object 的任務選單、Claude 選任務、跑真 rollout（每任務數分鐘）、
> 最後印出完成/中止。換指令或 `--suite libero_goal` 看不同任務。

## 預期輸出
- 任務選單（10 個 libero_object 任務的語言指令）
- agent 拆解 reasoning + 選的 task id
- `execute(task N: ...) -> 成功`
- `✅ 完成`

## 已知雷
- **agent 選錯任務 / 對不到** → prompt 已把任務選單塞給它；對不到會回 needs_clarification。
- **rollout 很慢** → 正常（T4 ~3min/task），subprocess 每次重載 policy 故更慢；demo 用單一指令。
- **import libero 又問路徑** → 新 session 沒跑 step 4，補跑即可。
