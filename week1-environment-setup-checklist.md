# Week 1 — 環境設定 Checklist
## 切到 Claude Code 後的第一份執行指南

> 用途：放進 Claude Code 專案，作為 Week 1「環境跑通」的執行清單。
> 對應 spec：§5 Week 1 ＋ §10 立即可開始的第一步。
> 成功訊號：**能在模擬裡看到虛擬手臂執行一次動作。** 達到這個，Week 1 就算過關。

---

## ⚠️ 給 Claude Code 的開場指示（把這段貼給它）

```
我要開始一個 LeRobot + SmolVLA 的純模擬 side project。專案的完整設計在這幾份檔案裡：
- physical-ai-agent-project-spec.md（整體藍圖）
- agent-task-decomposition-prompt.md（agent 拆解設計）
- query-visual-judgment-optimization.md（視覺判斷優化）

請先讀完這三份，再協助我完成 Week 1 環境設定。

重要：環境設定的具體指令、套件版本、相依關係請以「當下的 LeRobot 官方 repo 與
官方文件」為準，不要憑記憶給可能過期的指令。先去 github.com/huggingface/lerobot
查證最新的安裝方式，再帶我一步步做。
```

> 這段話是關鍵：強制 Claude Code 對著「當下的官方 repo」落地，而不是用可能過期的記憶。

---

## 1. 前置確認（開工前先盤點）

- [ ] **算力**：確認你要用哪個環境跑
  - 選項 A：雲端 GPU（推薦，省設定）——L4 / A100 等級
  - 選項 B：官方 Colab Notebook（零成本起步，最省事）
  - 選項 C：本機（若你有夠力的 GPU；Mac 也可跑 SmolVLA 但較慢）
- [ ] **Python 環境**：建議用 conda / venv 開乾淨環境，避免污染既有專案
- [ ] **Hugging Face 帳號**：之後要下載 `lerobot/smolvla_base` 與資料集，先確認能登入

---

## 2. 核心安裝步驟（讓 Claude Code 對照官方落地）

> 不寫死指令，列出「要完成的事」+「已知的雷」，讓 Claude Code 查官方給你當下正確版本。

- [ ] **取得 LeRobot**：clone 官方 repo（github.com/huggingface/lerobot）
- [ ] **安裝 LeRobot + smolvla 相依**：官方通常有 extras 安裝法（如帶 smolvla 的 optional dependencies），請 Claude Code 查當下正確寫法
- [ ] **安裝模擬環境相依**：你要的模擬 benchmark（如 LIBERO / Meta-World 類）對應的套件
- [ ] **驗證安裝**：能 import lerobot、能列出可用的 policy 與模擬環境

### 🔴 已知的雷（Week 1 最常卡這幾個）
1. **ffmpeg / 影像編解碼**：LeRobot 處理影像資料常需要 ffmpeg 或 TorchCodec，版本不對會報錯。→ 對策：照官方指定版本裝，別用系統預設。
2. **PyTorch 與 CUDA 版本對應**：GPU 跑不起來十之八九是這裡。→ 對策：先確認你的 CUDA 版本，再裝對應的 PyTorch。
3. **模擬環境的額外系統依賴**：某些模擬器需要額外的系統函式庫（如 OpenGL 相關）。→ 對策：用官方 Docker image 可繞過大半。
4. **首次下載模型/資料集卡住**：網路或 HF 認證問題。→ 對策：先 `huggingface-cli login`。

> 強烈建議：若選項允許，**優先用官方提供的 Docker image 或 Colab**，能繞過上面大半的環境地獄。

---

## 3. 跑通官方範例（Week 1 的核心目標）

- [ ] **載入 `lerobot/smolvla_base`**：確認模型能成功下載並載入
- [ ] **載入官方範例資料集**：`lerobot/svla_so101_pickplace` 或模擬內建資料集
- [ ] **跑一次模擬評估 / 推論**：讓 smolvla_base 在模擬環境裡執行，產生動作
- [ ] **🎯 成功訊號**：在模擬畫面裡，**看到虛擬手臂動起來、執行一次抓取嘗試**

> 達到這個成功訊號，Week 1 就過關了。不要求成功率、不要求漂亮，只要求「會動」。

---

## 4. Week 1 收尾（為 Week 2 鋪路）

- [ ] 記錄你實際用的：環境（雲/Colab/本機）、套件版本、踩過的雷與解法
  → 這份記錄之後直接變成你拆解文的「環境建置」素材，也方便 debug 回溯
- [ ] 確認下一步（Week 2）的入口：找到官方「用 svla_so101_pickplace 微調 smolvla_base」
      的訓練指令在哪（通常在官方 examples / 文件）

---

## 5. 心法提醒（來自 spec，別忘了）

- **先用官方資料把管線跑通，再換自己的**：Week 1-2 的鐵律。先確認「載入→執行→評估」
  整條路順，之後出問題才分得清是資料還是設定的錯。
- **不要太早改架構**：先讓預設設定正常運作，別一上來就調參數、改 policy。
- **卡關超過合理時間就換路徑**：本機裝不起來 → 立刻轉 Colab / Docker，別在環境地獄耗一整週。

---

### 如果 Week 1 嚴重卡關（退路）
環境設定是公認最容易勸退人的一關。若你在本機卡超過一兩個晚上：
1. 直接改用官方 Colab Notebook，幾乎零設定，先讓你看到「手臂會動」建立信心
2. 把「完整環境」留到確認整個 project 值得做下去之後再搞
3. 記住：Week 1 的唯一目標是「看到手臂動起來」，不是「建好完美環境」
