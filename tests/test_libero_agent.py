"""task-level 編排測試：agent 用 execute(task) 技能驅動（LIBERO 的本機替身）。

FakeLiberoSkill 不依賴 lerobot，Mac 上可跑；驗證 agent.py 的 execute 路徑
（成功序列、失敗重試、abort），與 mock 的 pick/place 路徑共用同一套編排骨架。
"""

from agent.brain import Brain, FakeClient
from agent.agent import Agent
from agent.libero_skills import LiberoSkillInterface
from agent.rollout_engine import exec_output_dir, RolloutOutcome
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


def test_higher_max_retries_allows_more_attempts():
    """--max-retries 拉高重試上限 → 更多次獨立 rollout 嘗試（demo 在 20% 難任務上
    用來提高『失敗→重試→救回』的出現機率）。Agent 本就支援 max_retries，這支鎖住
    該行為，避免日後改壞 flag 的效果。"""
    skills = FakeLiberoSkill()
    skills.fail_next_execute = 3  # 前 3 次 rollout 失敗
    # 預設 max_retries=2（3 試）會在此 abort；拉到 3（4 試）則第 4 次成功救回。
    agent = Agent(brain=Brain(FakeClient(_plan("task A"))), skills=skills, max_retries=3)
    result = agent.run("做 task A")
    assert result.status == "completed"
    assert len(skills.executed) == 4  # 1 + 3 retries


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


# ── LiberoSkillInterface 注入式測試 ──────────────────────────────────────────

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
    assert engine.runs == [(4, False, 1)]
