"""L2 技能介面測試：pick/place/query，query 依 mode 路由到座標真值或語意。"""

from agent.world import MockWorld
from agent.skills import SkillInterface


def make_skills() -> SkillInterface:
    return SkillInterface(MockWorld.default_scene())


def test_pick_success_returns_ok():
    skills = make_skills()
    result = skills.pick("red cube")
    assert result.ok is True


def test_pick_unknown_object_returns_not_ok():
    skills = make_skills()
    result = skills.pick("green pyramid")
    assert result.ok is False


def test_place_success_returns_ok():
    skills = make_skills()
    skills.pick("red cube")
    result = skills.place("zone_A")
    assert result.ok is True


def test_query_spatial_in_zone_true_after_place():
    skills = make_skills()
    skills.pick("red cube")
    skills.place("zone_A")
    assert skills.query("red cube 是否在 zone_A？", mode="spatial") is True


def test_query_spatial_in_zone_false_before_place():
    skills = make_skills()
    assert skills.query("red cube 是否在 zone_A？", mode="spatial") is False


def test_query_spatial_held_true_after_pick():
    skills = make_skills()
    assert skills.query("red cube 是否已被夾起？", mode="spatial") is False
    skills.pick("red cube")
    assert skills.query("red cube 是否已被夾起？", mode="spatial") is True


def test_query_semantic_lists_objects_and_colors():
    skills = make_skills()
    answer = skills.query("場上有哪些物件？各是什麼顏色？", mode="semantic")
    assert isinstance(answer, str)
    assert "red" in answer and "blue" in answer
