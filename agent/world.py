"""L1 替身：MockWorld。

用純 Python 狀態模擬「桌面 + 彩色方塊 + 放置區」，供 L2/L3 在無真實 sim、
無 GPU 的條件下驗證 agent 拆解與閉環邏輯。座標皆為真值，未來換 LIBERO
時只替換本層與 skills.py，L3 不動。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldObject:
    id: str
    color: str
    pos: tuple[float, float]
    held: bool = False


@dataclass
class Zone:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, pos: tuple[float, float]) -> bool:
        x, y = pos
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class MockWorld:
    def __init__(self, objects: list[WorldObject], zones: dict[str, Zone]) -> None:
        self._objects = objects
        self.zones = zones
        self.fail_next_pick = 0  # >0 時，接下來這麼多次 pick 會被強制失敗

    @classmethod
    def default_scene(cls) -> MockWorld:
        objects = [
            WorldObject(id="red_cube", color="red", pos=(0.1, 0.1)),
            WorldObject(id="blue_cube", color="blue", pos=(0.1, -0.1)),
        ]
        zones = {
            "zone_A": Zone("zone_A", 0.2, 0.4, 0.2, 0.4),
            "box_left": Zone("box_left", -0.6, -0.4, -0.1, 0.1),
            "box_right": Zone("box_right", 0.4, 0.6, -0.1, 0.1),
        }
        return cls(objects, zones)

    def list_objects(self) -> list[WorldObject]:
        return list(self._objects)

    @property
    def held_object(self) -> WorldObject | None:
        for obj in self._objects:
            if obj.held:
                return obj
        return None

    def find_object(self, query: str) -> WorldObject | None:
        """以自然語言描述比對物件（依顏色或 id 關鍵字）。"""
        normalized = query.lower().replace(" ", "_")
        for obj in self._objects:
            if obj.id in normalized or obj.color in query.lower():
                return obj
        return None

    def pick(self, query: str) -> bool:
        if self.fail_next_pick > 0:
            self.fail_next_pick -= 1
            return False
        if self.held_object is not None:
            return False
        obj = self.find_object(query)
        if obj is None:
            return False
        obj.held = True
        return True

    def place(self, target: str) -> bool:
        obj = self.held_object
        if obj is None:
            return False
        zone = self.zones.get(target)
        if zone is None:
            return False
        obj.pos = zone.center
        obj.held = False
        return True

    def is_in_zone(self, query: str, zone_name: str) -> bool:
        obj = self.find_object(query)
        zone = self.zones.get(zone_name)
        if obj is None or zone is None:
            return False
        return zone.contains(obj.pos)

    def get_position(self, query: str) -> tuple[float, float] | None:
        obj = self.find_object(query)
        return obj.pos if obj else None
