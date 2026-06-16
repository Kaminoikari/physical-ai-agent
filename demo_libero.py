"""Kaggle 上的端到端 demo：一句人話 → Claude 拆解成 LIBERO 任務 → 真 rollout → 閉環。

需在 Kaggle（Linux + GPU）執行，且需設好 ANTHROPIC_API_KEY 與 MUJOCO_GL=egl。
用法：
    python demo_libero.py "把字母湯罐頭收進籃子"
    python demo_libero.py "把湯罐頭和番茄醬都收進籃子" --suite libero_object
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from agent.agent import Agent
from agent.brain import AnthropicClient, Brain
from agent.libero_prompts import build_system_prompt
from agent.libero_skills import LiberoSkillInterface

_STATUS_LABEL = {
    "completed": "✅ 完成",
    "out_of_scope": "⛔ 超出任務範圍",
    "needs_clarification": "❓ 需要澄清",
    "aborted": "🛑 中止",
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Physical AI agent × 真 LIBERO demo")
    parser.add_argument("instruction", help="自然語言指令")
    parser.add_argument("--suite", default="libero_object", help="LIBERO suite")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="agent 大腦模型")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="execute 失敗的重試上限（共 max_retries+1 次嘗試）。"
        "在 libero_10 等低成功率任務上調高，可提高『失敗→重試→救回』的出現機率。",
    )
    args = parser.parse_args()

    print(f"🗣  指令：{args.instruction}")
    print(f"🔧 載入 LIBERO suite={args.suite} 與 SmolVLA checkpoint…")
    skills = LiberoSkillInterface(suite=args.suite)

    print("📋 可用任務選單：")
    for task_id, language in skills.available_tasks():
        print(f"   {task_id}: {language}")

    system_prompt = build_system_prompt(skills.available_tasks())
    agent = Agent(
        brain=Brain(AnthropicClient(model=args.model), system_prompt=system_prompt),
        skills=skills,
        max_retries=args.max_retries,
    )

    print("\n🤖 agent 規劃並執行中（每個任務 rollout 約數分鐘）…\n")
    result = agent.run(args.instruction)

    print("—— 執行紀錄 ——")
    for line in result.log:
        print(f"  • {line}")
    print(f"\n{_STATUS_LABEL.get(result.status, result.status)}：{result.message}")
    print(f"🎬 各次 rollout 影片（含失敗/成功）：{skills.output_root}/")


if __name__ == "__main__":
    main()
