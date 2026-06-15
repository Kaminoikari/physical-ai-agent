"""MockWorld（L1 替身）機制測試。"""

from agent.world import MockWorld


def test_default_scene_has_demo_objects_and_zones():
    world = MockWorld.default_scene()
    ids = {obj.id for obj in world.list_objects()}
    assert {"red_cube", "blue_cube"} <= ids
    assert {"zone_A", "box_left", "box_right"} <= set(world.zones)


def test_pick_by_color_holds_object():
    world = MockWorld.default_scene()
    assert world.pick("red cube") is True
    assert world.held_object is not None
    assert world.held_object.id == "red_cube"
    assert world.find_object("red cube").held is True


def test_pick_nonexistent_object_fails():
    world = MockWorld.default_scene()
    assert world.pick("green pyramid") is False
    assert world.held_object is None


def test_place_moves_held_object_into_zone_and_releases():
    world = MockWorld.default_scene()
    world.pick("red cube")
    assert world.place("zone_A") is True
    assert world.held_object is None
    assert world.is_in_zone("red cube", "zone_A") is True


def test_place_with_empty_gripper_fails():
    world = MockWorld.default_scene()
    assert world.place("zone_A") is False


def test_is_in_zone_is_false_before_placing():
    world = MockWorld.default_scene()
    assert world.is_in_zone("red cube", "zone_A") is False


def test_fail_next_pick_injection_forces_failure_then_succeeds():
    world = MockWorld.default_scene()
    world.fail_next_pick = 1
    assert world.pick("red cube") is False  # 被注入的失敗
    assert world.held_object is None
    assert world.pick("red cube") is True  # 下一次恢復正常
    assert world.held_object.id == "red_cube"
