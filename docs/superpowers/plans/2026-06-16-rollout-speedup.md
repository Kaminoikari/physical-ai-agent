# 自我迭代閉環加速 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `execute` 每 task 重載 policy 的數量級開銷，並讓同一引擎能一鍵切到 GR00T N1.5，全程不破壞真 ground-truth 成敗的可信度。

**Architecture:** 在 `LiberoSkillInterface` 與「怎麼跑 rollout」之間切一條可注入的 `RolloutEngine` 接縫。`InProcessRolloutEngine` 用官方 `eval_policy` 把 policy/processor/env 常駐複用（#1+#3）；builder 參數化成 `(policy_type, policy_path)` 以切 SmolVLA/GR00T（#4）；`agent.py` 加完成記憶與 plan-only 模式（#2）。碰 lerobot 的程式碼壓到最薄、其餘純邏輯本機 mock 可測。

**Tech Stack:** Python、lerobot（`eval_policy`/`make_policy`/`make_env`/processors）、LIBERO、pytest、Anthropic SDK（既有）。

---

## File Structure

- **Create** `agent/rollout_engine.py` — `RolloutOutcome`、`PolicyBundle`、`RolloutEngine`/`PolicyEnvBuilder` 協定、`InProcessRolloutEngine`、`SubprocessRolloutEngine`、`LerobotPolicyEnvBuilder`。
- **Modify** `agent/libero_skills.py` — `LiberoSkillInterface` 注入 `engine`/`tasks`/`save_video`/`n_episodes`，`execute` 委派引擎並套 `success_threshold`。
- **Modify** `agent/agent.py` — `run()` 加完成記憶 + `plan_only` 模式旗標。
- **Create** `tests/test_rollout_engine.py` — 引擎/builder 的 mock 測。
- **Modify** `tests/test_libero_agent.py` — `execute` 委派 + 門檻判定（注入 fake 引擎與 fake task 清單）。
- **Modify** `tests/test_agent.py`（或既有 agent 測檔）— 完成記憶 + plan-only 模式。
- **Create** `bench_rollout.py` — Kaggle 速度實測 + 成敗 parity 腳本（含 `--policy groot`）。

> 註：`RolloutOutcome` 只帶 `pc_success: float` 與 `video_path`，**門檻判定留在 interface**（spec §5.1）。這比 spec 草稿的 `success: bool` 欄位乾淨：引擎只回原始數值，成敗語意單一歸屬 interface。

---

## Task 1: RolloutOutcome 與協定骨架

**Files:**
- Create: `agent/rollout_engine.py`
- Test: `tests/test_rollout_engine.py`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_rollout_engine.py
from agent.rollout_engine import RolloutOutcome


def test_rollout_outcome_holds_pc_success_and_optional_video():
    outcome = RolloutOutcome(pc_success=75.0)
    assert outcome.pc_success == 75.0
    assert outcome.video_path is None
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.rollout_engine'`

- [ ] **Step 3: 寫最小實作**

```python
# agent/rollout_engine.py
"""L1/L2 加速接縫：把『怎麼跑 rollout』從 LiberoSkillInterface 抽出成可注入引擎。

三實作：InProcessRolloutEngine（常駐複用，#1+#3）、SubprocessRolloutEngine（現行 baseline）、
測試用 FakeRolloutEngine（在測試檔，不在此）。所有碰 lerobot 的呼叫皆 lazy import，本機 import 此檔不失敗。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RolloutOutcome:
    """引擎只回原始 pc_success；成敗門檻由 LiberoSkillInterface 判定。"""

    pc_success: float
    video_path: str | None = None


@dataclass
class PolicyBundle:
    """常駐複用的整組：policy + eval_policy 需要的四條 processor pipeline。"""

    policy: Any
    env_preprocessor: Any
    env_postprocessor: Any
    preprocessor: Any
    postprocessor: Any


class PolicyEnvBuilder(Protocol):
    policy_type: str
    policy_path: str

    def build_policy(self) -> PolicyBundle: ...

    def build_env(self, task_id: int) -> Any: ...


class RolloutEngine(Protocol):
    def run(self, task_id: int, *, save_video: bool, n_episodes: int) -> RolloutOutcome: ...
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/rollout_engine.py tests/test_rollout_engine.py
git commit -m "feat: 新增 RolloutOutcome 與 rollout 引擎協定骨架

把『怎麼跑 rollout』抽成可注入接縫,為常駐複用與 GR00T 切換鋪路。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: InProcessRolloutEngine — 常駐複用邏輯（核心 #1/#3）

**Files:**
- Modify: `agent/rollout_engine.py`
- Test: `tests/test_rollout_engine.py`

設計重點：`eval_fn` 與 `builder` 皆可注入 → 整支引擎在本機用 fake 即可測，零 lerobot。

- [ ] **Step 1: 寫失敗測試（builder 只建一次、env 按 task 快取、save_video 旗標、回傳值）**

```python
# tests/test_rollout_engine.py  （append）
from agent.rollout_engine import InProcessRolloutEngine, PolicyBundle


class FakeBuilder:
    policy_type = "smolvla"
    policy_path = "fake/checkpoint"

    def __init__(self):
        self.build_policy_calls = 0
        self.build_env_calls: list[int] = []

    def build_policy(self) -> PolicyBundle:
        self.build_policy_calls += 1
        return PolicyBundle("POLICY", "envpre", "envpost", "pre", "post")

    def build_env(self, task_id: int):
        self.build_env_calls.append(task_id)
        return f"ENV{task_id}"


class FakeEval:
    """記錄被呼叫的引數，回固定 info（仿 eval_policy 回傳結構）。"""

    def __init__(self, pc_success=80.0):
        self.calls: list[dict] = []
        self._pc = pc_success

    def __call__(self, env, policy, env_pre, env_post, pre, post, *, n_episodes,
                 max_episodes_rendered, videos_dir):
        self.calls.append(
            {"env": env, "policy": policy, "n_episodes": n_episodes,
             "max_episodes_rendered": max_episodes_rendered, "videos_dir": videos_dir}
        )
        return {"aggregated": {"pc_success": self._pc}, "video_paths": []}


def test_inprocess_builds_policy_once_across_runs():
    builder = FakeBuilder()
    engine = InProcessRolloutEngine(builder, eval_fn=FakeEval())
    for _ in range(3):
        engine.run(0, save_video=False, n_episodes=1)
    assert builder.build_policy_calls == 1


def test_inprocess_caches_env_per_task_id():
    builder = FakeBuilder()
    engine = InProcessRolloutEngine(builder, eval_fn=FakeEval())
    engine.run(0, save_video=False, n_episodes=1)
    engine.run(0, save_video=False, n_episodes=1)
    engine.run(2, save_video=False, n_episodes=1)
    assert builder.build_env_calls == [0, 2]  # task 0 第二次命中快取、不重建


def test_inprocess_save_video_flag_controls_render_and_videos_dir():
    fake_eval = FakeEval()
    engine = InProcessRolloutEngine(FakeBuilder(), eval_fn=fake_eval, videos_dir="/tmp/v",
                                    max_render_episodes=1)
    engine.run(0, save_video=False, n_episodes=1)
    engine.run(0, save_video=True, n_episodes=1)
    assert fake_eval.calls[0]["max_episodes_rendered"] == 0
    assert fake_eval.calls[0]["videos_dir"] is None
    assert fake_eval.calls[1]["max_episodes_rendered"] == 1
    assert fake_eval.calls[1]["videos_dir"] == "/tmp/v"


def test_inprocess_returns_pc_success_from_eval_info():
    engine = InProcessRolloutEngine(FakeBuilder(), eval_fn=FakeEval(pc_success=42.0))
    outcome = engine.run(0, save_video=False, n_episodes=1)
    assert outcome.pc_success == 42.0
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -k inprocess -v`
Expected: FAIL — `ImportError: cannot import name 'InProcessRolloutEngine'`

- [ ] **Step 3: 寫實作**

```python
# agent/rollout_engine.py  （append）
class InProcessRolloutEngine:
    """policy/processor 全 session 建一次；env 按 task_id lazy 建並快取。內呼官方 eval_policy。"""

    def __init__(self, builder: PolicyEnvBuilder, *, eval_fn=None,
                 videos_dir: str | None = None, max_render_episodes: int = 1) -> None:
        self._builder = builder
        self._eval_fn = eval_fn
        self._videos_dir = videos_dir
        self._max_render_episodes = max_render_episodes
        self._bundle: PolicyBundle | None = None
        self._envs: dict[int, Any] = {}

    def _ensure_eval_fn(self):
        if self._eval_fn is None:
            from lerobot.scripts.lerobot_eval import eval_policy  # lazy：Kaggle-only
            self._eval_fn = eval_policy
        return self._eval_fn

    def run(self, task_id: int, *, save_video: bool, n_episodes: int) -> RolloutOutcome:
        if self._bundle is None:
            self._bundle = self._builder.build_policy()
        if task_id not in self._envs:
            self._envs[task_id] = self._builder.build_env(task_id)
        eval_fn = self._ensure_eval_fn()
        b = self._bundle
        info = eval_fn(
            self._envs[task_id], b.policy,
            b.env_preprocessor, b.env_postprocessor, b.preprocessor, b.postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=(self._max_render_episodes if save_video else 0),
            videos_dir=(self._videos_dir if save_video else None),
        )
        videos = info.get("video_paths") or []
        return RolloutOutcome(
            pc_success=float(info["aggregated"]["pc_success"]),
            video_path=videos[0] if videos else None,
        )
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -k inprocess -v`
Expected: PASS（4 個）

- [ ] **Step 5: Commit**

```bash
git add agent/rollout_engine.py tests/test_rollout_engine.py
git commit -m "feat: InProcessRolloutEngine 常駐複用 policy/env

policy 全 session 建一次、env 按 task 快取,消除每 task 重載;save_video 預設關閉省渲染。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: SubprocessRolloutEngine — 把現行 _run_eval 搬成 baseline 引擎

**Files:**
- Modify: `agent/rollout_engine.py`
- Test: `tests/test_rollout_engine.py`

- [ ] **Step 1: 寫失敗測試（mock subprocess + info 檔，驗解析 pc_success 與遞增 seq）**

```python
# tests/test_rollout_engine.py  （append）
import json

from agent.rollout_engine import SubprocessRolloutEngine


def test_subprocess_engine_parses_pc_success(tmp_path, monkeypatch):
    class FakeProc:
        returncode = 0
        stderr = ""

    def fake_run(cmd, env, capture_output, text):
        # cmd 中的 --output_dir=<dir> → 在該 dir 寫 eval_info.json
        out_dir = [a.split("=", 1)[1] for a in cmd if a.startswith("--output_dir=")][0]
        with open(f"{out_dir}/eval_info.json", "w") as f:
            json.dump({"overall": {"pc_success": 60.0}}, f)
        return FakeProc()

    monkeypatch.setattr("agent.rollout_engine.subprocess.run", fake_run)
    engine = SubprocessRolloutEngine(suite="libero_object", checkpoint="ckpt",
                                     device="cuda", output_root=str(tmp_path))
    outcome = engine.run(0, save_video=False, n_episodes=1)
    assert outcome.pc_success == 60.0
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -k subprocess -v`
Expected: FAIL — `ImportError: cannot import name 'SubprocessRolloutEngine'`

- [ ] **Step 3: 寫實作（搬移 libero_skills 既有 _run_eval 邏輯）**

```python
# agent/rollout_engine.py  （append，並在檔頭補 import）
import json
import os
import subprocess


def _exec_output_dir(output_root: str, suite: str, task_id: int, seq: int) -> str:
    return os.path.join(output_root, suite, f"task{task_id}", f"run{seq}")


class SubprocessRolloutEngine:
    """現行行為：每次 subprocess 叫 lerobot-eval（會重載 policy）。保留作 Kaggle 速度 baseline。"""

    def __init__(self, *, suite: str, checkpoint: str, device: str,
                 output_root: str) -> None:
        self._suite = suite
        self._checkpoint = checkpoint
        self._device = device
        self._output_root = output_root
        self._exec_seq = 0

    def run(self, task_id: int, *, save_video: bool, n_episodes: int) -> RolloutOutcome:
        out_dir = _exec_output_dir(self._output_root, self._suite, task_id, self._exec_seq)
        self._exec_seq += 1
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            "lerobot-eval",
            f"--policy.path={self._checkpoint}",
            f"--policy.device={self._device}",
            "--env.type=libero",
            f"--env.task={self._suite}",
            f"--env.task_ids=[{task_id}]",
            "--eval.batch_size=1",
            f"--eval.n_episodes={n_episodes}",
            "--env.max_parallel_tasks=1",
            f"--output_dir={out_dir}",
        ]
        env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        info_path = os.path.join(out_dir, "eval_info.json")
        if proc.returncode != 0 or not os.path.exists(info_path):
            raise RuntimeError(f"lerobot-eval 失敗（task {task_id}）：\n{proc.stderr[-2000:]}")
        with open(info_path) as f:
            info = json.load(f)
        return RolloutOutcome(pc_success=float(info["overall"]["pc_success"]))
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -k subprocess -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/rollout_engine.py tests/test_rollout_engine.py
git commit -m "refactor: 把 _run_eval 搬成 SubprocessRolloutEngine

保留現行 subprocess 行為作為 Kaggle 速度/parity 的 baseline 引擎。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: LerobotPolicyEnvBuilder — 碰 lerobot 的薄膠水（#4 GR00T 接縫）

**Files:**
- Modify: `agent/rollout_engine.py`
- Test: `tests/test_rollout_engine.py`

膠水的 lerobot 呼叫無法本機測（lazy import）；本機只測「參數化正確、預設 SmolVLA、可換 GR00T」。
`build_policy`/`build_env` 的接線**對齊官方 `eval_main()`**，在 Kaggle 驗。

- [ ] **Step 1: 寫失敗測試（只驗參數，不觸發 lerobot）**

```python
# tests/test_rollout_engine.py  （append）
from agent.rollout_engine import LerobotPolicyEnvBuilder


def test_builder_defaults_to_smolvla():
    builder = LerobotPolicyEnvBuilder()
    assert builder.policy_type == "smolvla"
    assert builder.policy_path == "HuggingFaceVLA/smolvla_libero"


def test_builder_can_target_groot():
    builder = LerobotPolicyEnvBuilder(policy_type="groot", policy_path="nvidia/GR00T-N1.5-3B")
    assert builder.policy_type == "groot"
    assert builder.policy_path == "nvidia/GR00T-N1.5-3B"
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -k builder -v`
Expected: FAIL — `ImportError: cannot import name 'LerobotPolicyEnvBuilder'`

- [ ] **Step 3: 寫實作（接線對齊官方 eval_main：make_policy / make_pre_post_processors / make_env_pre_post_processors / make_env）**

```python
# agent/rollout_engine.py  （append）
class LerobotPolicyEnvBuilder:
    """碰 lerobot 的薄膠水：建 policy + 四 processor + 各 task 的 env，全部 lazy import。

    接線對齊官方 lerobot_eval.eval_main()。policy_type/policy_path 參數化 → 換 GR00T 只改這兩個值。
    cfg 物件的精確欄位依安裝的 lerobot 版本在 Kaggle 上微調（這是本檔唯一 Kaggle-only 的區塊）。
    """

    def __init__(self, *, policy_type: str = "smolvla",
                 policy_path: str = "HuggingFaceVLA/smolvla_libero",
                 suite: str = "libero_object", device: str = "cuda",
                 batch_size: int = 1) -> None:
        self.policy_type = policy_type
        self.policy_path = policy_path
        self.suite = suite
        self.device = device
        self.batch_size = batch_size
        self._policy_cfg = None
        self._env_cfg = None

    def _cfgs(self):
        if self._policy_cfg is None:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.envs.configs import LiberoEnv

            self._policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
            self._policy_cfg.type = self.policy_type
            self._policy_cfg.device = self.device
            self._policy_cfg.pretrained_path = self.policy_path
            self._env_cfg = LiberoEnv(task=self.suite)
        return self._policy_cfg, self._env_cfg

    def build_policy(self) -> PolicyBundle:
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.envs.factory import make_env_pre_post_processors

        policy_cfg, env_cfg = self._cfgs()
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )
        return PolicyBundle(policy, env_preprocessor, env_postprocessor,
                            preprocessor, postprocessor)

    def build_env(self, task_id: int):
        from lerobot.envs.factory import make_env

        _, env_cfg = self._cfgs()
        env_cfg.task_ids = [task_id]
        envs = make_env(env_cfg, n_envs=self.batch_size, use_async_envs=False,
                        trust_remote_code=True)
        # make_env 對 libero 回 {suite: {task_id: VectorEnv}}；取出單一 task 的 VectorEnv
        if isinstance(envs, dict):
            suite_envs = next(iter(envs.values()))
            return suite_envs[task_id] if task_id in suite_envs else next(iter(suite_envs.values()))
        return envs
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_rollout_engine.py -k builder -v`
Expected: PASS（2 個；不觸發 lerobot import）

- [ ] **Step 5: Commit**

```bash
git add agent/rollout_engine.py tests/test_rollout_engine.py
git commit -m "feat: LerobotPolicyEnvBuilder——參數化 policy 以一鍵切 GR00T

接線對齊官方 eval_main;policy_type/policy_path 參數化,SmolVLA 為預設、GR00T 為 opt-in。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: LiberoSkillInterface 改委派引擎 + 注入 task 清單

**Files:**
- Modify: `agent/libero_skills.py`
- Test: `tests/test_libero_agent.py`

讓 interface 在本機可測：`tasks` 與 `engine` 皆可注入；省略時才走 lerobot/預設引擎（Kaggle）。

- [ ] **Step 1: 寫失敗測試（注入 fake 引擎 + fake task 清單，驗門檻判定）**

```python
# tests/test_libero_agent.py  （append）
from agent.libero_skills import LiberoSkillInterface
from agent.rollout_engine import RolloutOutcome


class FakeEngine:
    def __init__(self, pc_success):
        self._pc = pc_success
        self.runs: list[tuple] = []

    def run(self, task_id, *, save_video, n_episodes):
        self.runs.append((task_id, save_video, n_episodes))
        return RolloutOutcome(pc_success=self._pc)


FAKE_TASKS = [(0, "pick up the alphabet soup"), (4, "pick up the ketchup")]


def test_execute_ok_when_pc_success_meets_threshold():
    skills = LiberoSkillInterface(tasks=FAKE_TASKS, engine=FakeEngine(80.0),
                                  success_threshold=50.0)
    result = skills.execute("0")
    assert result.ok is True


def test_execute_fails_when_pc_success_below_threshold():
    skills = LiberoSkillInterface(tasks=FAKE_TASKS, engine=FakeEngine(20.0),
                                  success_threshold=50.0)
    result = skills.execute("0")
    assert result.ok is False


def test_execute_delegates_task_id_and_flags_to_engine():
    engine = FakeEngine(80.0)
    skills = LiberoSkillInterface(tasks=FAKE_TASKS, engine=engine, n_episodes=1)
    skills.execute("4")
    assert engine.runs == [(4, False, 1)]  # save_video 預設 False（快迴圈不寫 mp4）
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_libero_agent.py -k execute -v`
Expected: FAIL — `TypeError`（`__init__` 尚不收 `tasks`/`engine`）

- [ ] **Step 3: 改實作（`__init__` 收 `tasks`/`engine`/`save_video`，`execute` 委派引擎）**

```python
# agent/libero_skills.py — 取代 __init__、execute、刪除 _run_eval
from agent.rollout_engine import InProcessRolloutEngine, LerobotPolicyEnvBuilder


class LiberoSkillInterface:
    def __init__(
        self,
        suite: str = "libero_object",
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        n_episodes: int = 1,
        success_threshold: float = 50.0,
        save_video: bool = False,
        tasks: list[tuple[int, str]] | None = None,
        engine=None,
    ) -> None:
        self.suite = suite
        self.checkpoint = checkpoint
        self.device = device
        self.n_episodes = n_episodes
        self.success_threshold = success_threshold
        self.save_video = save_video
        self._tasks = tasks if tasks is not None else self._load_task_list()
        self._engine = engine if engine is not None else InProcessRolloutEngine(
            LerobotPolicyEnvBuilder(policy_path=checkpoint, suite=suite, device=device)
        )

    def execute(self, task: str) -> SkillResult:
        task_id = self._resolve_task_id(task)
        language = self._tasks[task_id][1]
        outcome = self._engine.run(task_id, save_video=self.save_video,
                                   n_episodes=self.n_episodes)
        ok = outcome.pc_success >= self.success_threshold
        return SkillResult(
            ok=ok,
            detail=f"execute(task {task_id}: {language}) -> {'成功' if ok else '失敗'}",
        )
```

> `_load_task_list`、`available_tasks`、`_resolve_task_id`、`query` 不動。刪掉舊的 `_run_eval` 與 `exec_output_dir`/`DEFAULT_OUTPUT_ROOT`（已搬到 rollout_engine）。`_resolve_task_id` 用 `self._tasks` 的索引；注意 fake 的 task id 與 list 索引一致（0、4 → 用 dict 對映較穩，見下方修正）。

- [ ] **Step 3b: 修正 task_id 對映（fake 清單 id≠索引）**

```python
# agent/libero_skills.py — _resolve_task_id 後、execute 取 language 改用 dict
    def execute(self, task: str) -> SkillResult:
        task_id = self._resolve_task_id(task)
        language = dict(self._tasks)[task_id]
        outcome = self._engine.run(task_id, save_video=self.save_video,
                                   n_episodes=self.n_episodes)
        ok = outcome.pc_success >= self.success_threshold
        return SkillResult(
            ok=ok,
            detail=f"execute(task {task_id}: {language}) -> {'成功' if ok else '失敗'}",
        )
```

- [ ] **Step 4: 跑測試確認通過 + 既有測不破**

Run: `.venv/bin/python -m pytest tests/test_libero_agent.py -v`
Expected: PASS（新 3 + 既有）

- [ ] **Step 5: Commit**

```bash
git add agent/libero_skills.py tests/test_libero_agent.py
git commit -m "refactor: LiberoSkillInterface 委派 RolloutEngine,門檻判定留在 interface

execute 改呼引擎,成敗語意單一歸屬 interface;tasks/engine 可注入故本機可測。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: agent.py 完成記憶（#2）

**Files:**
- Modify: `agent/agent.py`
- Test: `tests/test_agent.py`（無則新建）

- [ ] **Step 1: 寫失敗測試（plan 重複 execute 同一 task → 第二次跳過）**

```python
# tests/test_agent.py  （append；沿用既有 FakeBrain/FakeSkills 風格，無則新增下方版本）
from agent.agent import Agent
from agent.schemas import Plan, Step, SkillResult


class _Brain:
    def __init__(self, plan):
        self._plan = plan

    def decompose(self, instruction, observation=None):
        return self._plan


class _Skills:
    def __init__(self):
        self.execute_calls: list[str] = []

    def execute(self, arg):
        self.execute_calls.append(arg)
        return SkillResult(ok=True, detail=f"execute({arg}) -> 成功")

    def query(self, q, mode="semantic"):
        return ""


def test_completed_memo_skips_repeated_successful_execute():
    plan = Plan(reasoning="r", in_scope=True, needs_clarification=False,
                clarification_question="", steps=[Step("execute", "4"), Step("execute", "4")])
    skills = _Skills()
    agent = Agent(_Brain(plan), skills)
    agent.run("先收番茄醬再收番茄醬", assume_success=True)
    assert skills.execute_calls == ["4"]  # 第二個 execute 4 被完成記憶跳過
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_agent.py -k memo -v`
Expected: FAIL — `execute_calls == ["4", "4"]`（尚無記憶）

- [ ] **Step 3: 改實作（run 內維護 completed set；execute 步驟成功後標記、命中即跳）**

```python
# agent/agent.py — run() 內，迴圈前加 completed；step 迴圈內 execute 分支加守衛
    def run(self, instruction: str, assume_success: bool = False) -> RunResult:
        log: list[str] = []
        observation: str | None = None
        self._current_object: str | None = None
        completed: set[str] = set()  # 已成功的 execute arg，replan 重排則跳過

        for _ in range(self._max_iters):
            plan = self._brain.decompose(instruction, observation)
            log.append(f"拆解：{plan.reasoning}")

            if not plan.in_scope:
                return RunResult("out_of_scope", plan.reasoning, log)
            if plan.needs_clarification:
                return RunResult("needs_clarification", plan.clarification_question, log)
            if not plan.steps:
                return RunResult("completed", "無動作需執行", log)

            action_taken = False
            for step in plan.steps:
                if step.skill == "abort":
                    return RunResult("aborted", step.arg, log)
                if step.skill == "query":
                    observation = str(self._skills.query(step.arg, mode="semantic"))
                    log.append(f"觀察：{observation}")
                    continue
                if step.skill == "execute" and step.arg in completed:
                    log.append(f"跳過 execute({step.arg})：已完成（completed memo）")
                    continue
                if step.skill in ("pick", "place", "execute"):
                    action_taken = True
                    if not self._execute_with_retry(step, assume_success, log):
                        return RunResult("aborted", f"{step.skill}({step.arg}) 連續失敗", log)
                    if step.skill == "execute":
                        completed.add(step.arg)

            if action_taken:
                return RunResult("completed", "任務完成", log)

        return RunResult("aborted", "超過迭代上限", log)
```

- [ ] **Step 4: 跑測試確認通過 + 既有 agent 測不破**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/agent.py tests/test_agent.py
git commit -m "feat: agent 完成記憶——已成功子任務 replan 時跳過

同一 episode 內已放好的物件維持放好,避免重跑 rollout;確定性安全,不快取隨機成敗。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: agent.py plan-only 模式（#2 快迴圈）

**Files:**
- Modify: `agent/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: 寫失敗測試（plan_only=True → 不呼叫 execute，回傳含計畫）**

```python
# tests/test_agent.py  （append）
def test_plan_only_mode_does_not_execute():
    plan = Plan(reasoning="把醬料收起來", in_scope=True, needs_clarification=False,
                clarification_question="", steps=[Step("execute", "4")])
    skills = _Skills()
    agent = Agent(_Brain(plan), skills)
    result = agent.run("把所有醬料收起來", plan_only=True)
    assert skills.execute_calls == []          # 快迴圈不跑 rollout
    assert result.status == "planned"
    assert any("execute" in line for line in result.log)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_agent.py -k plan_only -v`
Expected: FAIL — `TypeError: run() got an unexpected keyword argument 'plan_only'`

- [ ] **Step 3: 改實作（run 收 plan_only；為真時 decompose 後印計畫即回，不執行）**

```python
# agent/agent.py — run 簽名加 plan_only；in_scope/clarification 檢查後、執行前插入分支
    def run(self, instruction: str, assume_success: bool = False,
            plan_only: bool = False) -> RunResult:
        log: list[str] = []
        observation: str | None = None
        self._current_object: str | None = None
        completed: set[str] = set()

        for _ in range(self._max_iters):
            plan = self._brain.decompose(instruction, observation)
            log.append(f"拆解：{plan.reasoning}")

            if not plan.in_scope:
                return RunResult("out_of_scope", plan.reasoning, log)
            if plan.needs_clarification:
                return RunResult("needs_clarification", plan.clarification_question, log)
            if not plan.steps:
                return RunResult("completed", "無動作需執行", log)

            if plan_only:
                for index, step in enumerate(plan.steps, start=1):
                    log.append(f"  計畫 {index}. {step.skill}({step.arg})")
                return RunResult("planned", "計畫產出（未跑 rollout）", log)

            # ……以下執行迴圈同 Task 6……
```

> 註：執行迴圈（`action_taken` 那段）維持 Task 6 的版本，只在其前插入 `if plan_only:` 早退分支。`RunResult.status` 新增合法值 `"planned"`。

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/agent.py tests/test_agent.py
git commit -m "feat: agent plan-only 模式——快迴圈只拆解不跑 rollout

讓自我迭代的思考層以 API 速度迭代計畫,昂貴 rollout 只在計畫穩定後批次驗收。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: bench_rollout.py — Kaggle 速度實測 + 成敗 parity（含 GR00T）

**Files:**
- Create: `bench_rollout.py`

無單元測（Kaggle-only 腳本，仿 `check_menu.py`）；驗收靠 Step 2 的本機 import 檢查（頂層不可觸發 lerobot）。

- [ ] **Step 1: 寫腳本**

```python
# bench_rollout.py
"""Kaggle 實測：① in-process vs subprocess 的每 task 牆鐘時間；② 兩引擎成敗 parity。

用法（Kaggle，repo 根目錄，GPU）：
  python bench_rollout.py                 # SmolVLA，跑前幾個 task
  python bench_rollout.py --policy groot  # GR00T N1.5（需 Ampere+ GPU，見 spec 附錄 A）
只在 Kaggle/租用 GPU 跑；頂層僅 import 類別（lerobot 於引擎內 lazy import）。
"""

from __future__ import annotations

import argparse
import time

from agent.rollout_engine import (
    InProcessRolloutEngine,
    LerobotPolicyEnvBuilder,
    SubprocessRolloutEngine,
)

POLICY_SPECS = {
    "smolvla": ("smolvla", "HuggingFaceVLA/smolvla_libero"),
    "groot": ("groot", "nvidia/GR00T-N1.5-3B"),
}


def _time_engine(label, engine, task_ids):
    print(f"\n== {label} ==")
    results = {}
    for task_id in task_ids:
        start = time.perf_counter()
        outcome = engine.run(task_id, save_video=False, n_episodes=1)
        elapsed = time.perf_counter() - start
        results[task_id] = outcome.pc_success
        print(f"  task {task_id}: pc_success={outcome.pc_success:.1f}  {elapsed:.1f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=list(POLICY_SPECS), default="smolvla")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="0,1,2")
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()

    task_ids = [int(t) for t in args.tasks.split(",")]
    policy_type, policy_path = POLICY_SPECS[args.policy]

    in_proc = InProcessRolloutEngine(
        LerobotPolicyEnvBuilder(policy_type=policy_type, policy_path=policy_path,
                                suite=args.suite)
    )
    in_results = _time_engine(f"in-process [{args.policy}]", in_proc, task_ids)

    # baseline 僅對 SmolVLA 有意義（subprocess 走 lerobot-eval 預設 checkpoint）
    if args.policy == "smolvla":
        sub = SubprocessRolloutEngine(suite=args.suite, checkpoint=policy_path,
                                      device="cuda", output_root="/kaggle/working/bench")
        sub_results = _time_engine("subprocess [baseline]", sub, task_ids)

        print("\n== 成敗 parity（門檻 %.0f）==" % args.threshold)
        for task_id in task_ids:
            a = in_results[task_id] >= args.threshold
            b = sub_results[task_id] >= args.threshold
            mark = "✅" if a == b else "❌"
            print(f"  task {task_id}: in={a} sub={b} {mark}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 本機驗證頂層 import 安全（不觸發 lerobot）**

Run: `.venv/bin/python -c "import bench_rollout; print('import ok')"`
Expected: `import ok`（頂層只 import 類別，lerobot 在引擎內 lazy）

- [ ] **Step 3: Commit**

```bash
git add bench_rollout.py
git commit -m "feat: bench_rollout.py——Kaggle 速度實測 + 成敗 parity（含 GR00T 切換）

量 in-process vs subprocess 牆鐘時間,並斷言兩引擎成敗判定一致以守 ground-truth 可信度。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: 文件——bench 用法與全測綠

**Files:**
- Modify: `README.md`
- Modify: `docs/demo-results.md`

結果數字待 Kaggle 跑完才填；本任務只接上用法說明與測試數，不放佔位數字。

- [ ] **Step 1: 跑全測確認綠**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS（既有 42 + 新增約 10）

- [ ] **Step 2: README 補 bench 用法 + 更新測試數**

在 `README.md` 的「跑測試」段下方加一行 bench 用法；把兩處 `42 tests` 更新為實際數字（跑 `pytest -q` 末行的數）。

```markdown
### 量測 rollout 加速（Kaggle）

```bash
python bench_rollout.py                 # in-process vs subprocess 牆鐘時間 + 成敗 parity
python bench_rollout.py --policy groot  # 切 GR00T N1.5（需 Ampere+ GPU，見 spec 附錄 A）
```
```

- [ ] **Step 3: demo-results 加「⑦ 加速實測」段（先放方法、結果待填）**

在 `docs/demo-results.md` 末加一段，寫清楚量法與「待 Kaggle 實測填入」，不放假數字：

```markdown
## ⑦ rollout 加速實測（in-process vs subprocess）

方法：`bench_rollout.py` 對同批 task 跑兩引擎，量每 task 牆鐘時間並斷言成敗 parity。
預期：in-process 因 policy 常駐，每 task 從 ~3 min（subprocess 重載）降到數秒級。
> 實測數字待在 Kaggle/租用 GPU 跑出後填入（含 SmolVLA；GR00T 視 GPU 取得情況）。
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/demo-results.md
git commit -m "docs: 接上 bench_rollout 用法與加速實測段（數字待 Kaggle 填）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review（已對 spec 逐項核）

- **#1 常駐**：Task 2（InProcess 複用）+ Task 5（interface 委派）✅
- **#2 plan-only + 完成記憶**：Task 7 + Task 6 ✅
- **#3 微優化**：Task 2 的 save_video=False 預設（不寫 mp4）+ Task 4 batch_size ✅
- **#4 GR00T 接縫**：Task 4 參數化 builder + Task 8 `--policy groot` ✅
- **§6 測試**：本機 mock（Task 2/4/5/6/7）+ Kaggle parity（Task 8）✅
- **型別一致**：`RolloutOutcome.pc_success`、`RolloutEngine.run(task_id, *, save_video, n_episodes)`、`PolicyBundle` 四 processor、`Plan/Step` 沿用既有 schema——全程一致。
- **退路**：若 `eval_policy` 不能常駐複用，只改 `InProcessRolloutEngine.run` 內呼 `rollout()`，接縫與門檻不動（spec §7）。
- **無佔位數字**：demo-results 結果段明標「待 Kaggle 填」，非假數字。
