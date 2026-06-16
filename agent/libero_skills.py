"""L1+L2（真實版）：LiberoSkillInterface — task-level 技能，驅動真 LIBERO + SmolVLA。

與 mock 的 SkillInterface 同樣被 agent.py 呼叫，但技能粒度是「整段 task 的 rollout」：
- available_tasks(): 列出該 suite 每個任務的自然語言指令（= agent 的技能選單）
- execute(task):    跑一次 LIBERO rollout，回傳 LIBERO 的 ground-truth success

⚠️ 只能在 Linux + GPU（Kaggle）跑，無法在 Mac 本機執行。所有重依賴皆 lazy import，
   讓本機 import 此模組不致失敗（但 execute/available_tasks 實際呼叫需在 Kaggle）。

execute 現在委派給可注入的 RolloutEngine（預設 InProcessRolloutEngine，
policy 常駐複用、速度較快）；subprocess 版保留為 SubprocessRolloutEngine
baseline，在需要隔離或相容性測試時使用。
"""

from __future__ import annotations

from agent.rollout_engine import InProcessRolloutEngine, LerobotPolicyEnvBuilder, RolloutEngine
from agent.schemas import SkillResult

DEFAULT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"


class LiberoSkillInterface:
    def __init__(
        self,
        suite: str = "libero_object",
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        n_episodes: int = 1,
        success_threshold: float = 50.0,  # pc_success(%) 達標即視為成功
        save_video: bool = False,
        tasks: list[tuple[int, str]] | None = None,
        engine: RolloutEngine | None = None,
    ) -> None:
        self.suite = suite
        self.checkpoint = checkpoint
        self.device = device
        self.n_episodes = n_episodes
        self.success_threshold = success_threshold
        self.save_video = save_video
        self._tasks = tasks if tasks is not None else self._load_task_list()
        self._language_by_id = dict(self._tasks)
        self._engine = (
            engine
            if engine is not None
            else InProcessRolloutEngine(
                LerobotPolicyEnvBuilder(
                    policy_path=checkpoint, suite=suite, device=device
                )
            )
        )

    def _load_task_list(self) -> list[tuple[int, str]]:
        from libero.libero import benchmark  # lazy：只在 Kaggle 需要

        suite_cls = benchmark.get_benchmark_dict()[self.suite]
        task_suite = suite_cls()
        return [(i, task_suite.get_task(i).language) for i in range(task_suite.n_tasks)]

    def available_tasks(self) -> list[tuple[int, str]]:
        return list(self._tasks)

    def _resolve_task_id(self, task: str) -> int:
        """agent 給的 arg 可能是 task id（推薦）或語言字串；兩種都解析。"""
        task = task.strip()
        if task.isdigit():
            return int(task)
        # 退而求其次：用語言字串做寬鬆比對
        lowered = task.lower()
        for task_id, language in self._tasks:
            if lowered in language.lower() or language.lower() in lowered:
                return task_id
        raise ValueError(f"無法對應到任何 LIBERO 任務：{task!r}")

    def execute(self, task: str) -> SkillResult:
        task_id = self._resolve_task_id(task)
        language = self._language_by_id[task_id]
        outcome = self._engine.run(task_id, save_video=self.save_video,
                                   n_episodes=self.n_episodes)
        ok = outcome.pc_success >= self.success_threshold
        return SkillResult(
            ok=ok,
            detail=f"execute(task {task_id}: {language}) -> {'成功' if ok else '失敗'}",
        )

    # 與 mock 對稱：語意查詢（這裡用任務清單回答「有哪些任務」）
    def query(self, question: str, mode: str = "semantic") -> str:
        return "可用任務：" + "；".join(f"{i}={lang}" for i, lang in self._tasks)
