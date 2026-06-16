"""L1/L2 加速接縫：把『怎麼跑 rollout』從 LiberoSkillInterface 抽出成可注入引擎。

三實作：InProcessRolloutEngine（常駐複用，#1+#3）、SubprocessRolloutEngine（現行 baseline）、
測試用 FakeRolloutEngine（在測試檔，不在此）。所有碰 lerobot 的呼叫皆 lazy import，本機 import 此檔不失敗。
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
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


# ── SubprocessRolloutEngine ──────────────────────────────────────────────────

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


# ── LerobotPolicyEnvBuilder ──────────────────────────────────────────────────

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

    def _ensure_cfgs(self) -> tuple[Any, Any]:
        if self._policy_cfg is None:
            from lerobot.configs.policies import PreTrainedConfig  # lazy：Kaggle-only
            from lerobot.envs.configs import LiberoEnv              # lazy：Kaggle-only

            self._policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
            self._policy_cfg.type = self.policy_type
            self._policy_cfg.device = self.device
            self._policy_cfg.pretrained_path = self.policy_path
            self._env_cfg = LiberoEnv(task=self.suite)
        return self._policy_cfg, self._env_cfg

    def build_policy(self) -> PolicyBundle:
        from lerobot.policies.factory import make_policy, make_pre_post_processors  # lazy
        from lerobot.envs.factory import make_env_pre_post_processors               # lazy

        policy_cfg, env_cfg = self._ensure_cfgs()
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

        _, shared_env_cfg = self._ensure_cfgs()
        env_cfg = copy.deepcopy(shared_env_cfg)  # 每次 local 複製，勿污染共用快取
        env_cfg.task_ids = [task_id]
        envs = make_env(env_cfg, n_envs=self.batch_size, use_async_envs=False,
                        trust_remote_code=True)
        # make_env 對 libero 回 {suite: {task_id: VectorEnv}}；取出單一 task 的 VectorEnv（lerobot 型別，Kaggle-only）
        if isinstance(envs, dict):
            suite_envs = next(iter(envs.values()))
            return suite_envs[task_id] if task_id in suite_envs else next(iter(suite_envs.values()))
        return envs
