"""task-level 編排測試：agent 用 execute(task) 技能驅動（LIBERO 的本機替身）。

FakeLiberoSkill 不依賴 lerobot，Mac 上可跑；驗證 agent.py 的 execute 路徑
（成功序列、失敗重試、abort），與 mock 的 pick/place 路徑共用同一套編排骨架。
"""

from agent.brain import Brain, FakeClient
from agent.agent import Agent
from agent.libero_skills import exec_output_dir
from agent.schemas import SkillResult


class FakeLiberoSkill:
    """in-memory 的 task-level 技能替身：execute(task) 回腳本化 success。"""

    def __init__(self) -> None:
        self.fail_next_execute = 0
        self.executed: list[str] = []

    def execute(self, task: str) -> SkillResult:
        self.executed.append(task)
        if self.fail_next_execute > 0:
            self.fail_next_execute -= 1
            return SkillResult(ok=False, detail=f"execute({task}) -> 失敗")
        return SkillResult(ok=True, detail=f"execute({task}) -> 成功")


def _plan(*tasks: str) -> str:
    steps = ",".join(f'{{"skill":"execute","arg":"{t}"}}' for t in tasks)
    return (
        '{"reasoning":"對應到 LIBERO 任務","in_scope":true,"needs_clarification":false,'
        f'"clarification_question":"","plan":[{steps}]}}'
    )


def make_agent(skills, client) -> Agent:
    return Agent(brain=Brain(client), skills=skills)


def test_single_execute_completes():
    skills = FakeLiberoSkill()
    agent = make_agent(skills, FakeClient(_plan("pick up the soup and place it in the basket")))
    result = agent.run("把湯罐頭收進籃子")
    assert result.status == "completed"
    assert skills.executed == ["pick up the soup and place it in the basket"]


def test_multi_task_sequence_completes():
    skills = FakeLiberoSkill()
    agent = make_agent(skills, FakeClient(_plan("task A", "task B")))
    result = agent.run("依序做兩件事")
    assert result.status == "completed"
    assert skills.executed == ["task A", "task B"]


def test_execute_failure_then_retry_succeeds():
    skills = FakeLiberoSkill()
    skills.fail_next_execute = 1
    agent = make_agent(skills, FakeClient(_plan("task A")))
    result = agent.run("做 task A")
    assert result.status == "completed"
    assert any("重試" in line for line in result.log)


def test_persistent_execute_failure_aborts():
    skills = FakeLiberoSkill()
    skills.fail_next_execute = 9
    agent = make_agent(skills, FakeClient(_plan("task A")))
    result = agent.run("做 task A")
    assert result.status == "aborted"


def test_execute_retry_log_says_rollout_not_verify():
    """execute 是 task-level rollout，沒有座標驗證那步；重試訊息要誠實講 rollout 失敗。"""
    skills = FakeLiberoSkill()
    skills.fail_next_execute = 1
    agent = make_agent(skills, FakeClient(_plan("task A")))
    result = agent.run("做 task A")
    retry_lines = [line for line in result.log if "重試" in line]
    assert retry_lines
    assert all("rollout" in line for line in retry_lines)
    assert all("驗證" not in line for line in retry_lines)


def test_exec_output_dir_is_persistent_and_unique():
    """rollout 輸出要寫到持久、可辨識、每次 attempt 不互相覆蓋的路徑，
    這樣失敗/成功的 mp4 才留得住（取代原本用過即丟的 tempfile）。"""
    root = "/kaggle/working/libero_exec"
    first = exec_output_dir(root, "libero_10", 0, 0)
    second = exec_output_dir(root, "libero_10", 0, 1)
    assert first.startswith(root)          # 持久根目錄，非 /tmp
    assert "/tmp" not in first             # 不再用 tempfile
    assert "libero_10" in first            # 可辨識 suite
    assert "task0" in first                # 可辨識 task
    assert first != second                 # 每次 attempt 各自一個目錄，不覆蓋
