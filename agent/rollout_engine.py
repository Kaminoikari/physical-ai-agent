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
