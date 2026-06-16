"""軸 A：plan-only 拆解驗證——只驗 agent 的拆解計畫，不跑 rollout。

放大「為什麼需要 L3」的論點：語意分群、排除/否定、3+ 物件、排序這類語言推理，是
編排大腦的活，VLA policy 一次只吃一個原子任務、做不到。plan-only 把這層單獨拎出來，
零 GPU、零 sim、秒回——只呼叫 brain.decompose() 一次，把計畫印出來。

刻意不碰 skills.execute()：plan-only 不存在「成功/失敗」結果，只有「計畫」。執行層的
成敗由情境①〜⑤的真 rollout 證據負責，兩者邊界分明、不過度宣稱。
"""

from __future__ import annotations

from agent.brain import Brain
from agent.schemas import Plan


def decompose_only(brain: Brain, instruction: str) -> Plan:
    """只拆解、不執行：呼叫一次 brain.decompose()。"""
    return brain.decompose(instruction)


def format_plan(plan: Plan, tasks: list[tuple[int, str]]) -> list[str]:
    """把 Plan 轉成人類可讀的行。

    execute 步驟標成「計畫（未跑 rollout）」並把 task_id 映回語言字串——絕不出現
    「成功/失敗」，因為 plan-only 不宣稱執行結果。out_of_scope / needs_clarification
    各自格式化。
    """
    language_by_id = {task_id: language for task_id, language in tasks}

    lines = [f"拆解：{plan.reasoning}"]
    if not plan.in_scope:
        lines.append(f"⛔ 超出範圍：{plan.reasoning}")
        return lines
    if plan.needs_clarification:
        lines.append(f"❓ 需要澄清：{plan.clarification_question}")
        return lines

    lines.append("📋 計畫（未跑 rollout，只驗拆解）：")
    for index, step in enumerate(plan.steps, start=1):
        if step.skill == "execute":
            language = _language_for(step.arg, language_by_id)
            lines.append(f"  {index}. execute(task {step.arg}: {language})")
        else:
            lines.append(f"  {index}. {step.skill}({step.arg})")
    return lines


def _language_for(arg: str, language_by_id: dict[int, str]) -> str:
    """把 execute 的 arg（task_id 數字字串）映回語言；非數字或越界則原樣回傳。"""
    if arg.strip().isdigit():
        return language_by_id.get(int(arg), arg)
    return arg
