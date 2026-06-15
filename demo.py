"""CLI demo：一句自然語言指令 → Claude 拆解 → 在 mock world 執行 → 閉環回饋。

用法：
    .venv/bin/python demo.py "把紅色方塊放到 A 區"
    .venv/bin/python demo.py "紅色方塊放 box_left，藍色方塊放 box_right"
    .venv/bin/python demo.py "把零件擦乾淨"            # 超出技能範圍
    .venv/bin/python demo.py "..." --assume-success    # 跳過閉環驗證（Week 3 模式）
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from agent.agent import Agent
from agent.brain import AnthropicClient, Brain
from agent.skills import SkillInterface
from agent.world import MockWorld

_STATUS_LABEL = {
    "completed": "✅ 完成",
    "out_of_scope": "⛔ 超出技能範圍",
    "needs_clarification": "❓ 需要澄清",
    "aborted": "🛑 中止",
}


def _describe_scene(world: MockWorld) -> str:
    return "、".join(f"{o.id}({o.color})@{o.pos}" for o in world.list_objects())


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Physical AI agent demo（純模擬）")
    parser.add_argument("instruction", help="自然語言指令")
    parser.add_argument("--assume-success", action="store_true", help="跳過閉環驗證")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="agent 大腦模型")
    args = parser.parse_args()

    world = MockWorld.default_scene()
    agent = Agent(brain=Brain(AnthropicClient(model=args.model)), skills=SkillInterface(world))

    print(f"🗣  指令：{args.instruction}")
    print(f"🌍 初始場景：{_describe_scene(world)}")
    print(f"📦 可用區域：{', '.join(world.zones)}\n")

    result = agent.run(args.instruction, assume_success=args.assume_success)

    print("—— 執行紀錄 ——")
    for line in result.log:
        print(f"  • {line}")
    print(f"\n{_STATUS_LABEL.get(result.status, result.status)}：{result.message}")
    print(f"🌍 最終場景：{_describe_scene(world)}")


if __name__ == "__main__":
    main()
