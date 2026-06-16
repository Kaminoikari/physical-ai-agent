"""agent.Agent 單元測試：完成記憶（completed memo）等獨立行為。"""

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
