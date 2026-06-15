"""L2 技能介面：把 MockWorld 包成 agent 可呼叫的 pick/place/query。

簽章固定，是 L1 解耦邊界——未來 LiberoSkillInterface 用相同方法替換，L3 不變。
query 依 query-visual-judgment-optimization.md 設計：spatial 讀座標真值、
semantic 走語意（mock 階段由 world 狀態確定性作答，真實階段換 VLM）。
"""

from __future__ import annotations

from agent.schemas import SkillResult
from agent.world import MockWorld

_GRASP_KEYWORDS = ("夾", "抓", "握", "起", "pick", "grasp", "held", "hold")


class SkillInterface:
    def __init__(self, world: MockWorld) -> None:
        self.world = world

    def pick(self, obj: str) -> SkillResult:
        ok = self.world.pick(obj)
        return SkillResult(ok=ok, detail=f"pick({obj}) -> {'成功' if ok else '失敗'}")

    def place(self, target: str) -> SkillResult:
        ok = self.world.place(target)
        return SkillResult(ok=ok, detail=f"place({target}) -> {'成功' if ok else '失敗'}")

    def query(self, question: str, mode: str) -> bool | str:
        if mode == "spatial":
            return self._query_spatial(question)
        if mode == "semantic":
            return self._query_semantic(question)
        raise ValueError(f"未知的 query mode：{mode}")

    def _query_spatial(self, question: str) -> bool:
        """讀座標真值回答二元問題。優先判斷『在哪個 zone』，其次判斷『是否被夾起』。"""
        obj = self.world.find_object(question)
        if obj is None:
            return False
        zone_name = next((z for z in self.world.zones if z in question), None)
        if zone_name is not None:
            return self.world.is_in_zone(question, zone_name)
        if any(kw in question.lower() for kw in _GRASP_KEYWORDS):
            return obj.held
        return False

    def _query_semantic(self, question: str) -> str:
        objects = "、".join(f"{obj.id}({obj.color})" for obj in self.world.list_objects())
        zones = "、".join(self.world.zones)
        # 明示區域無顏色標示，agent 才能偵測「對應顏色的盒子」這類歧義並回報 needs_clarification
        return f"場上物件：{objects}；可用區域（無顏色標示）：{zones}"
