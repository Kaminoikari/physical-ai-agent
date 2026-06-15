"""L3 編排迴圈測試：端到端執行、短路、閉環重試/abort。全程 FakeClient。"""

from agent.world import MockWorld
from agent.skills import SkillInterface
from agent.brain import Brain, FakeClient
from agent.agent import Agent

PLAN_RED_TO_A = (
    '{"reasoning":"抓紅放A","in_scope":true,"needs_clarification":false,'
    '"clarification_question":"","plan":['
    '{"skill":"pick","arg":"red cube"},{"skill":"place","arg":"zone_A"}]}'
)
PLAN_OUT_OF_SCOPE = (
    '{"reasoning":"擦拭超範圍","in_scope":false,"needs_clarification":false,'
    '"clarification_question":"","plan":[]}'
)
PLAN_NEEDS_CLARIFY = (
    '{"reasoning":"盒子無顏色標示","in_scope":true,"needs_clarification":true,'
    '"clarification_question":"紅藍各放哪個盒子？","plan":[]}'
)


def make_agent(world: MockWorld, client) -> Agent:
    return Agent(brain=Brain(client), skills=SkillInterface(world))


def test_clear_instruction_completes_and_places_object():
    world = MockWorld.default_scene()
    agent = make_agent(world, FakeClient(PLAN_RED_TO_A))
    result = agent.run("把紅色方塊放到 A 區")
    assert result.status == "completed"
    assert world.is_in_zone("red cube", "zone_A") is True


def test_out_of_scope_short_circuits():
    world = MockWorld.default_scene()
    agent = make_agent(world, FakeClient(PLAN_OUT_OF_SCOPE))
    result = agent.run("幫我把零件擦乾淨")
    assert result.status == "out_of_scope"


def test_needs_clarification_short_circuits():
    world = MockWorld.default_scene()
    agent = make_agent(world, FakeClient(PLAN_NEEDS_CLARIFY))
    result = agent.run("把方塊分類到對應顏色的盒子")
    assert result.status == "needs_clarification"
    assert "盒子" in result.message


def test_pick_failure_then_retry_succeeds():
    world = MockWorld.default_scene()
    world.fail_next_pick = 1  # 第一次抓取失敗，重試應成功
    agent = make_agent(world, FakeClient(PLAN_RED_TO_A))
    result = agent.run("把紅色方塊放到 A 區")
    assert result.status == "completed"
    assert world.is_in_zone("red cube", "zone_A") is True
    assert any("重試" in line for line in result.log)


def test_persistent_pick_failure_aborts():
    world = MockWorld.default_scene()
    world.fail_next_pick = 9  # 持續失敗，超過重試上限應 abort
    agent = make_agent(world, FakeClient(PLAN_RED_TO_A))
    result = agent.run("把紅色方塊放到 A 區")
    assert result.status == "aborted"


def test_assume_success_mode_skips_verification():
    world = MockWorld.default_scene()
    world.fail_next_pick = 9  # 即使底層失敗，assume_success 也不驗證、照跑完
    agent = make_agent(world, FakeClient(PLAN_RED_TO_A))
    result = agent.run("把紅色方塊放到 A 區", assume_success=True)
    assert result.status == "completed"


def test_observe_then_act_replans_with_observation():
    world = MockWorld.default_scene()
    plan_observe_first = (
        '{"reasoning":"先看一眼","in_scope":true,"needs_clarification":false,'
        '"clarification_question":"","plan":[{"skill":"query","arg":"場上有哪些物件？"}]}'
    )
    client = FakeClient([plan_observe_first, PLAN_RED_TO_A])  # 第一輪只觀察，第二輪才行動
    agent = make_agent(world, client)
    result = agent.run("把紅色方塊放到 A 區")
    assert result.status == "completed"
    assert world.is_in_zone("red cube", "zone_A") is True
