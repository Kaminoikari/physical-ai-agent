"""L3 大腦測試：JSON 解析與 decompose，全程用 FakeClient（不打真 API）。"""

import pytest

from agent.brain import Brain, FakeClient, ParseError, parse_plan

VALID_JSON = """
{
  "reasoning": "先看一眼再抓放",
  "in_scope": true,
  "needs_clarification": false,
  "clarification_question": "",
  "plan": [
    {"skill": "query", "arg": "場上有哪些物件？"},
    {"skill": "pick", "arg": "red cube"},
    {"skill": "place", "arg": "zone_A"}
  ]
}
"""


def test_parse_plan_reads_steps():
    plan = parse_plan(VALID_JSON)
    assert plan.in_scope is True
    assert [s.skill for s in plan.steps] == ["query", "pick", "place"]
    assert plan.steps[1].arg == "red cube"


def test_parse_plan_strips_markdown_fences():
    raw = "```json\n" + VALID_JSON.strip() + "\n```"
    plan = parse_plan(raw)
    assert len(plan.steps) == 3


def test_parse_plan_malformed_json_raises():
    with pytest.raises(ParseError):
        parse_plan("這不是 JSON")


def test_parse_plan_missing_field_raises():
    with pytest.raises(ParseError):
        parse_plan('{"in_scope": true}')


def test_decompose_returns_plan_from_client():
    brain = Brain(FakeClient(VALID_JSON))
    plan = brain.decompose("把紅色方塊放到 A 區")
    assert plan.in_scope is True
    assert plan.steps[-1].skill == "place"


def test_decompose_out_of_scope():
    raw = '{"reasoning":"擦拭不在技能庫","in_scope":false,"needs_clarification":false,"clarification_question":"","plan":[]}'
    brain = Brain(FakeClient(raw))
    plan = brain.decompose("幫我把零件擦乾淨")
    assert plan.in_scope is False
    assert plan.steps == []


def test_decompose_needs_clarification():
    raw = '{"reasoning":"盒子無顏色標示","in_scope":true,"needs_clarification":true,"clarification_question":"紅藍各放哪個盒子？","plan":[]}'
    brain = Brain(FakeClient(raw))
    plan = brain.decompose("把方塊分類到對應顏色的盒子")
    assert plan.needs_clarification is True
    assert "盒子" in plan.clarification_question
