"""跨層共用型別。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillResult:
    """pick/place 等動作技能的執行結果。"""

    ok: bool
    detail: str = ""


@dataclass
class Step:
    """plan 裡的單一技能呼叫。"""

    skill: str  # "pick" | "place" | "query" | "abort"
    arg: str


@dataclass
class Plan:
    """LLM 拆解的結構化輸出（對應 agent-task-decomposition-prompt.md 的 JSON）。"""

    reasoning: str
    in_scope: bool
    needs_clarification: bool
    clarification_question: str
    steps: list[Step] = field(default_factory=list)
