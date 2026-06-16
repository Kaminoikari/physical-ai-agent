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

## 五種行為（真 LIBERO 上的證據）

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
「都」一個字 → 拆成兩個 execute 依序跑。這就是開頭那個反問的答案：**這件事 VLA 自己
做不到，因為它一次只吃一個原子任務。**（這裡 agent 把「番茄醬」對到 tomato sauce(5)，
但繁中口語「番茄醬」其實偏指 ketchup——這是個真歧義，我在下一節 plan-only 裡專門挖了它。）

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

**⑤ 真實失敗——重試到誠實放棄（真 L1 失敗）**
> 指令：put both the alphabet soup and the tomato sauce in the basket（`--suite libero_10`）
> ```
> 拆解：alphabet soup 和 tomato sauce 都放入籃子，對應任務 0。
> execute(task 0) -> 失敗
> rollout 失敗，重試（第 1 次）
> execute(task 0) -> 失敗
> rollout 失敗，重試（第 2 次）
> execute(task 0) -> 失敗
> 🛑 中止：連續失敗
> ```
前四種我都跑在 libero_object（單物件抓放、官方 checkpoint ~100%）。這第五種換到
`libero_10` 長程 suite——同一個 task 要在一次 rollout 裡依序收兩樣。我本來想拍一段「失敗
→重試→救回」的對照，於是把這個 task 跑了一輪又一輪，結果一次都沒成功：首次 probe 的
5 集裡有 1 集成功（看起來 ~20%），但接著 4 次 demo、共 15 次真 rollout 全敗（含把重試上限
開到 6 次的那輪）——加總起來 20 次只成功 1 次，這個 task 對 checkpoint 而言其實只有 **~5%**。
所以我把這個「失敗」如實寫進來：agent 每次都跑真 rollout、全敗、誠實 abort，失敗訊號是
LIBERO 的 ground-truth，不是我注入的假失敗（早期 mock 版用 `--fail-first` 腳本造假失敗，那是
演的；這裡是真的做不到）。它沒有假裝成功，也沒有無限重試。
> 註：「失敗→重試→成功」的救回路徑在程式上存在、也有確定性單元測試證明；但這顆 checkpoint
> 在這個 task 太弱，我跑了 4 輪都沒拍到救回，於是不硬湊——重試邏輯靠單元測試保證，真實世界
> 觀察到的結局就是一致的誠實 abort。能誠實說「我想拍卻拍不到、所以這樣記錄」，比一支僥倖的
> 影片更可信。

## 把「拆解」單獨拎出來驗：plan-only（零 GPU）

前五種都要跑真 rollout、要 GPU。但「為什麼需要 L3」的核心，其實落在**拆解這一層**——把人話
翻成一串原子任務。於是我加了個 `--plan-only` 路徑：只呼叫一次 `brain.decompose()`、印出計畫，
**不跑 rollout**，本機秒回、只花 Claude API。這讓我能用更難的語言去壓測拆解能力。

四句指令、同一個 system prompt（沒為它們改 prompt）：

- **語意分群**：「把**所有醬料**都收進籃子」→ agent 從 10 項物件裡挑出 4 項醬料
  `execute(2,3,4,5)`。「醬料」這個類別字面上不在任何 task 裡，得靠世界知識歸類——policy 不可能
  做到，它只懂「pick up the X」。
- **3+ 物件**：「字母湯、牛奶和柳橙汁都收」→ `execute(0,7,9)`，三步依序，證明多步拆解能擴展。
- **排序**：「先收番茄醬再收字母湯」→ 計畫順序就是先 ketchup(4)、後 alphabet soup(0)，尊重顯式排序。

最有意思的是**排除**那一句：「**除了**番茄醬，其他醬料都收」。否定語法判對了（從集合移除），
但它逼出兩個自然語言的真坑，比「全對」更值得寫：

1. **「番茄醬」是真歧義**。選單同時有 `ketchup`(4) 和 `tomato sauce`(5)，而繁中口語「番茄醬」
   通常指 ketchup。plan-only 裡 agent 一致把它判成 ketchup(4)；但情境②的真 rollout 裡同一個詞卻
   對到 tomato sauce(5)。**同一個 agent、同一個詞、不同情境給出不同答案**——這恰恰證明它是該反問的
   歧義，沒有單一「正確解」。我把情境②原本「正確對到 tomato sauce」的說法也一併誠實收回了。
2. **類別邊界會浮動**。agent 在排除題裡把 `cream cheese`（奶油乳酪）也算進「醬料」，分群題卻沒算。
   類別歸屬本身模糊——這不是 bug，是語言固有的，而它正是 L3 要面對、policy 碰不到的問題。

plan-only 只證明「拆解這一層」對，不證明 rollout 會成功（那由①〜⑤負責）。但它用幾乎零成本，
把「為什麼需要一層會分類、會處理否定與排序、還會在歧義前該反問的編排大腦」講得更滿。

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
- **成功率受官方 checkpoint 限制、未自己微調**：libero_object 單物件 ~100%；libero_10
  長程任務實測只有 ~5%（夠當真實失敗素材、不適合 happy-path）。沒有雙臂、堆疊。要拉高長程
  成功率得自行微調，本專案刻意只用官方 checkpoint。
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
