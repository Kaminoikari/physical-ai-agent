"""軸 A：plan-only 拆解驗證測試。

只驗 agent 的拆解計畫（Plan），不跑 rollout。全程 FakeClient（零 API），
驗證 format_plan 把 Plan 轉成人類可讀行的規則，以及 decompose_only 不執行技能。
"""

from agent.brain import Brain, FakeClient
from agent.libero_prompts import LIBERO_OBJECT_TASKS, build_system_prompt
from agent.plan_only import decompose_only, format_plan
from agent.schemas import Plan, Step

# 四種醬料：salad dressing(2)、bbq sauce(3)、ketchup(4)、tomato sauce(5)
_SAUCES = (2, 3, 4, 5)


def _exec_plan(*task_ids: int) -> str:
    steps = ",".join(f'{{"skill":"execute","arg":"{i}"}}' for i in task_ids)
    return (
        '{"reasoning":"對應到醬料任務","in_scope":true,"needs_clarification":false,'
        f'"clarification_question":"","plan":[{steps}]}}'
    )


def test_format_plan_labels_execute_as_plan_not_success():
    """plan-only 絕不宣稱執行結果：輸出必須標『計畫』、不得出現『成功』。"""
    plan = Plan(
        reasoning="收四種醬料",
        in_scope=True,
        needs_clarification=False,
        clarification_question="",
        steps=[Step(skill="execute", arg=str(i)) for i in _SAUCES],
    )
    lines = format_plan(plan, LIBERO_OBJECT_TASKS)
    joined = "\n".join(lines)
    assert "計畫" in joined
    assert "成功" not in joined
    assert "失敗" not in joined


def test_format_plan_maps_task_id_to_language():
    """execute 步驟要把 task_id 映回語言字串，不是只印數字。"""
    plan = Plan(
        reasoning="收番茄醬",
        in_scope=True,
        needs_clarification=False,
        clarification_question="",
        steps=[Step(skill="execute", arg="5")],
    )
    lines = format_plan(plan, LIBERO_OBJECT_TASKS)
    joined = "\n".join(lines)
    assert "tomato sauce" in joined
    assert "task 5" in joined


def test_grouping_instruction_formats_four_execute_lines():
    """語意分群：FakeClient 回 4 步 plan → 格式化出 4 行 execute。"""
    brain = Brain(FakeClient(_exec_plan(*_SAUCES)), system_prompt=build_system_prompt(LIBERO_OBJECT_TASKS))
    plan = decompose_only(brain, "把所有醬料都收進籃子")
    lines = format_plan(plan, LIBERO_OBJECT_TASKS)
    exec_lines = [ln for ln in lines if "execute(task" in ln]
    assert len(exec_lines) == 4


def test_out_of_scope_plan_formats_rejection_without_execute():
    plan = Plan(
        reasoning="擦桌子不在任務範圍內",
        in_scope=False,
        needs_clarification=False,
        clarification_question="",
        steps=[],
    )
    lines = format_plan(plan, LIBERO_OBJECT_TASKS)
    joined = "\n".join(lines)
    assert "超出範圍" in joined
    assert "擦桌子" in joined
    assert "execute(task" not in joined


def test_needs_clarification_plan_formats_question():
    plan = Plan(
        reasoning="指代不明",
        in_scope=True,
        needs_clarification=True,
        clarification_question="請問是哪樣東西？",
        steps=[],
    )
    lines = format_plan(plan, LIBERO_OBJECT_TASKS)
    joined = "\n".join(lines)
    assert "需要澄清" in joined
    assert "哪樣東西" in joined


def test_static_menu_matches_ten_libero_object_tasks():
    """靜態選單需與真 LIBERO libero_object 一致（10 項、id 連續）。"""
    assert len(LIBERO_OBJECT_TASKS) == 10
    assert [i for i, _ in LIBERO_OBJECT_TASKS] == list(range(10))
    assert LIBERO_OBJECT_TASKS[0][1] == "pick up the alphabet soup and place it in the basket"
