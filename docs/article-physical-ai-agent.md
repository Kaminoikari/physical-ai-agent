# 為什麼具身智慧需要一層 LLM 編排大腦——一個純模擬 side project 的拆解

## 從一個反問開始

近兩年的 VLA（Vision-Language-Action）模型，像 SmolVLA、π0、OpenVLA，已經能直接吃
「pick up the alphabet soup and place it in the basket」這種自然語言，端到端輸出馬達
動作。既然 policy 自己就懂語言了，那再疊一層 LLM agent 在上面，不是疊床架屋嗎？

這個 side project 就是為了回答這個反問。答案藏在三句指令裡：

- 「把字母湯**和番茄醬都**收進籃子」——VLA 一次 rollout 只完成一個原子任務，「都」要
  拆成兩次。
- 「幫我把**桌子擦乾淨**」——這動作不在任何訓練好的技能裡，該誠實拒絕，而不是硬抓。
- 「把**那個東西**收起來」——指代不明，該反問，而不是亂猜一個。

組合、排序、能力邊界、歧義澄清——這些都不是「把單一任務做好」的 policy 該負責的事，
而是一層**編排大腦**的事。我把這層叫 L3，這篇文章就是它的設計拆解。

## 三層架構與職責分離

```
L3  Agent 編排（Claude）      理解人話 → 拆解 → 排序 → 邊界/歧義判斷
L2  技能介面                  execute(task_id) / query
L1  Policy 執行（SmolVLA）     單一原子任務的抓放 rollout
```

職責切得很乾淨：

- **L1 只回答「這一個任務做成了沒」**，輸出 ground-truth success。它不懂「都」、不懂
  「擦桌子不會」、不懂指代。
- **L3 只負責「把人話翻譯成一串 L1 聽得懂的原子任務」**，外加判斷做不做得到、要不要
  問清楚。它不碰馬達、不渲染畫面。
- **L2 是中間的窄腰**：一組穩定的函式簽章（`execute`、`query`、`available_tasks`），
  讓 L3 完全不需要知道底下是真 sim 還是 mock。

這個切法帶來一個可驗證的工程主張：**換掉 L1，L3 一行不改。** 我用兩種 L1 實作驗證了
它——本機的 `MockWorld`，和雲端的真 LIBERO + SmolVLA。兩者共用同一份 `agent.py`。

## L3 的迴圈設計

`agent.py` 的核心是一個有限迭代的迴圈（`agent/agent.py`）：

```
decompose（交給大腦拆解）
  → 若 in_scope=false      → 回報「超出範圍」
  → 若 needs_clarification → 回報「需要澄清」並問問題
  → 否則逐步執行 plan：
       query 步驟     → 把觀察結果回灌，重新規劃一輪
       execute 步驟   → 跑，用回傳的 success 判成敗，失敗重試（上限 2 次），再失敗 abort
```

幾個刻意的設計決策：

**為什麼要 query 回灌重規劃？** 因為真實任務常常「要先看一眼才知道怎麼做」。把 query
的結果塞回下一輪 decompose，agent 就能「觀察 → 再決定」，而不是一次把計畫拍死。

**為什麼成敗要用 L1 回傳，而不是 agent 自己宣稱？** 早期版本有個 `assume_success`
旁路，先把端到端串起來。但那是假閉環。真版的 `execute` 回傳的是 LIBERO 的
ground-truth `pc_success`，agent 是拿真結果在判斷，不是自我感覺良好。

**為什麼要重試上限 + abort？** 因為 policy 會失敗。沒有上限就會無限重試；沒有 abort
就會假裝成功。把「會失敗」當一等公民處理，才像真系統。

## 從 mock 到真 sim：一個被現實修正的設計

這裡有個值得誠實講的轉折。最初的設計，技能粒度是 `pick` / `place`——我以為 L3 會把
「收進籃子」拆成「pick(罐頭) → place(籃子)」這種細顆粒動作序列。

接上真 LIBERO 後才發現：**SmolVLA + LIBERO 的官方 checkpoint 是 task-level 的**——
一次 language-conditioned rollout 就把「抓起罐頭並放進籃子」整段做完，沒有「只抓不放」
的子技能。

於是 L2 的技能粒度從 `pick/place` 改成 task-level 的 `execute(task)`。這也意味著我得
誠實收回一個過度承諾：原本想講「L3 一行不改就接上 LIBERO」，但實際上技能語意變了，
prompt 和技能介面要跟著調。**真正不變的是 L3 的編排邏輯（迴圈、驗證、重試、邊界判斷），
變的是 L2 技能的顆粒度。** 這個區分，比一句漂亮的「完全不改」更接近真相，也更有說服力。

## 四種行為（真 LIBERO 上的證據）

以下都在 Kaggle T4 上、用官方 `HuggingFaceVLA/smolvla_libero` 跑真 rollout 產出。
任務選單是 agent 啟動時動態列出的，prompt 不寫死任何任務。

**① 成功——語意對應 + 真閉環**
> 指令：把字母湯罐頭收進籃子
> ```
> 拆解：對應任務 0：pick up the alphabet soup and place it in the basket
> execute(task 0) -> 成功
> ✅ 完成
> ```
中文「字母湯」對到英文任務、execute 回傳 ground-truth success。

**② 多步——分解 + 排序（王牌）**
> 指令：把字母湯罐頭和番茄醬都收進籃子
> ```
> 拆解：字母湯（task 0）和番茄醬（task 5）都在清單內，依序執行即可。
> execute(task 0) -> 成功
> execute(task 5) -> 成功
> ✅ 完成
> ```
「都」一個字 → 拆成兩個 execute 依序跑。而且番茄醬正確對到 tomato sauce(5) 而非
ketchup(4)。這就是開頭那個反問的答案：**這件事 VLA 自己做不到，因為它一次只吃一個
原子任務。**

**③ 拒絕——能力邊界自覺**
> 指令：幫我把桌子擦乾淨
> ```
> 拆解：擦桌子不在任何已訓練任務範圍內，所有可用任務都是抓取放籃。
> ⛔ 超出任務範圍
> ```
關鍵：在 reasoning 階段就擋下，**完全沒跑 rollout**。不是抓到一半才失敗，是知道自己
不會就直說。

**④ 澄清——不幻覺、會反問**
> 指令：把那個東西收起來
> ```
> 拆解：「那個東西」沒指明是哪樣，需要先釐清。
> ❓ 需要澄清：請問您想收起來的是哪樣東西？例如 alphabet soup、…、orange juice？
> ```
指代不明時反問並列出選項，而不是賭一個。

## 工程實務：踩過的雷

純模擬不代表不踩雷。幾個真實發生的：

- **LIBERO 只在 Linux**：Mac 本機裝不了 `hf-libero`（`sys_platform=='linux'`），所以
  本機只能跑 mock 版，真 sim 推到 Kaggle。這反而逼出了「mock/real 雙軌、L1 解耦」的
  好設計。
- **LIBERO 首次 import 會互動式問路徑**，在 notebook 裡直接 EOFError。解法是事先寫好
  `~/.libero/config.yaml`，而且要用 `sysconfig` 推 site-packages 路徑、別寫死 Python
  版本。
- **Kaggle 的 PATH 陷阱**：`lerobot-eval` 這個 console script 裝在 `/usr/local/bin`，
  但 kernel 的 `os.environ["PATH"]` 和 `/bin/sh` 預設 PATH 都不含它，subprocess 直接
  `FileNotFoundError`。最穩的解法是用 `sys.executable -m lerobot.scripts.lerobot_eval`
  直接跑模組，繞開 PATH 與 shebang。
- **新 session 預設沒 GPU**：`nvidia-smi: command not found`、`torch 是 +cpu 版`，要在
  Settings 開 GPU 重啟重跑。

這些雷本身就是「把研究 demo 變成可重現流程」的成本，值得記錄。完整重現步驟在
`docs/week2-kaggle-*.md`。

## 誠實的限制

- **慢**：`execute` 經 subprocess 每次重載 policy（~3 min/task）。取捨是重用整條官方
  管線換穩定；可改為載一次 policy + 直呼 `eval_policy()` 加速。
- **單臂單物件**：LIBERO object suite 沒有雙臂、堆疊、長程任務。
- **Week 1 用 metaworld expert 腳本而非 SmolVLA**：SmolVLA 的 action/obs space 與
  metaworld 不相容，zero-shot 接不上；真正用 SmolVLA 驅動是 Week 2 的 LIBERO。
- **sim-to-real gap**：全程純模擬，沒上實體手臂。

把限制寫清楚，不是示弱，是讓「做到的部分」更可信。

## 這對 Physical AI 的意義

具身智慧的價值鏈裡，底層 policy（L1）正在快速商品化——開源 checkpoint 越來越多、
越來越好。真正難被取代、也最貼近產品的，是**上面那層會理解人話、會拆解、會判斷邊界、
會在不確定時反問的編排大腦**。它決定了一隻會抓東西的手臂，是不是一個能聽懂「幫我把桌
上兩樣東西收好、那個我不確定的先別動」的助手。

這個 side project 用最小的成本（純模擬、官方 checkpoint、不自己微調）把這層做出來、並
證明它與底層解耦——換 sim 不動編排邏輯。它不解決 sim-to-real，但它把「L3 該長什麼樣」
這個問題，從投影片變成了可以跑、可以驗、可以誠實討論限制的東西。
