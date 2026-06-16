"""L3 編排迴圈：把指令交給大腦拆解，逐步在 world 上執行，並用座標回饋閉環。

迴圈：decompose → 執行 plan → query 步驟回灌觀察後重新規劃；pick/place 步驟
執行後用 query(spatial) 驗證，失敗重試（上限 max_retries），再失敗 abort。
assume_success=True 時跳過驗證，先把端到端串起來（spec §5 的退路設計）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.brain import Brain
from agent.schemas import Step
from agent.skills import SkillInterface


@dataclass
class RunResult:
    status: str  # "completed" | "out_of_scope" | "needs_clarification" | "aborted"
    message: str
    log: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        brain: Brain,
        skills: SkillInterface,
        max_retries: int = 2,
        max_iters: int = 8,
    ) -> None:
        self._brain = brain
        self._skills = skills
        self._max_retries = max_retries
        self._max_iters = max_iters

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
            # 只有 query、沒有動作 → 帶著觀察結果再規劃一輪

        return RunResult("aborted", "超過迭代上限", log)

    def _execute_with_retry(self, step: Step, assume_success: bool, log: list[str]) -> bool:
        for attempt in range(self._max_retries + 1):
            result = self._run_action(step)
            log.append(result.detail)

            if assume_success:
                return True
            # execute（task-level）的成敗由技能 rollout 直接回傳；pick/place 另用座標驗證
            ok = result.ok if step.skill == "execute" else self._verify(step)
            if ok:
                return True
            if attempt < self._max_retries:
                reason = "rollout 失敗" if step.skill == "execute" else "驗證失敗"
                log.append(f"{reason}，重試 {step.skill}（第 {attempt + 1} 次）")
        return False

    def _run_action(self, step: Step):
        if step.skill == "pick":
            self._current_object = step.arg
            return self._skills.pick(step.arg)
        if step.skill == "place":
            return self._skills.place(step.arg)
        return self._skills.execute(step.arg)

    def _verify(self, step: Step) -> bool:
        if step.skill == "pick":
            return bool(self._skills.query(f"{step.arg} 是否已被夾起？", mode="spatial"))
        obj = self._current_object or step.arg
        return bool(self._skills.query(f"{obj} 是否在 {step.arg}？", mode="spatial"))
