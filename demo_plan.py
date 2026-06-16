"""軸 A plan-only demo：一句人話 → Claude 拆解成 LIBERO 任務序列 → 印計畫（不跑 rollout）。

只驗 agent 的拆解能力（語意分群／排除否定／3+ 物件／排序），零 GPU、零 sim、秒回，
只花 Claude API。本機（Mac）即可跑。

用法：
    python demo_plan.py "把所有醬料都收進籃子"                              # 本機：靜態選單
    python demo_plan.py "除了番茄醬，其他醬料都收進籃子"
    python demo_plan.py "把所有醬料都收進籃子" --live --suite libero_object  # Kaggle：真選單

需設好 ANTHROPIC_API_KEY（放在 gitignore 的 .env）。
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from agent.brain import AnthropicClient, Brain
from agent.libero_prompts import LIBERO_OBJECT_TASKS, build_system_prompt
from agent.plan_only import decompose_only, format_plan


def _resolve_tasks(suite: str, live: bool) -> list[tuple[int, str]]:
    """選單來源：--live 用真 LIBERO（Kaggle）；否則用靜態 snapshot（本機）。"""
    if live:
        from agent.libero_skills import LiberoSkillInterface  # lazy：本機不 import lerobot

        return LiberoSkillInterface(suite=suite).available_tasks()
    if suite != "libero_object":
        raise SystemExit(f"靜態選單只支援 libero_object；suite={suite} 請加 --live 用真選單")
    return LIBERO_OBJECT_TASKS


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Physical AI agent × plan-only 拆解驗證")
    parser.add_argument("instruction", help="自然語言指令")
    parser.add_argument("--suite", default="libero_object", help="LIBERO suite")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="agent 大腦模型")
    parser.add_argument(
        "--live",
        action="store_true",
        help="用真 LIBERO available_tasks() 取選單（需在 Kaggle）；預設用本機靜態 snapshot。",
    )
    args = parser.parse_args()

    tasks = _resolve_tasks(args.suite, args.live)

    print(f"🗣  指令：{args.instruction}")
    print(f"📋 可用任務選單（{'真 LIBERO' if args.live else '靜態 snapshot'}，suite={args.suite}）：")
    for task_id, language in tasks:
        print(f"   {task_id}: {language}")

    brain = Brain(AnthropicClient(model=args.model), system_prompt=build_system_prompt(tasks))
    print("\n🤖 agent 拆解中（不跑 rollout，只驗計畫）…\n")
    plan = decompose_only(brain, args.instruction)

    print("—— 拆解計畫 ——")
    for line in format_plan(plan, tasks):
        print(f"  {line}")


if __name__ == "__main__":
    main()
