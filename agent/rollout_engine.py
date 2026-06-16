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
