"""Kaggle 對照：靜態選單 snapshot 是否與真 LIBERO 動態選單一字不差。

plan-only 本機模式用 agent/libero_prompts.py 的 LIBERO_OBJECT_TASKS（手寫 snapshot），
這支腳本在 Kaggle 上用真 LiberoSkillInterface.available_tasks() 取動態選單，逐項比對，
證明 snapshot 沒有偏差（否則本機 plan-only 的 task_id 對映就不可信）。

只需 libero 套件可 import，不需 GPU（只讀任務 metadata、不跑 rollout）。
用法（Kaggle，repo 根目錄）：python check_menu.py
"""

from __future__ import annotations

from agent.libero_prompts import LIBERO_OBJECT_TASKS
from agent.libero_skills import LiberoSkillInterface


def main() -> None:
    static = list(LIBERO_OBJECT_TASKS)
    live = list(LiberoSkillInterface(suite="libero_object").available_tasks())

    print(f"靜態 snapshot：{len(static)} 項")
    print(f"真 LIBERO     ：{len(live)} 項")

    if static == live:
        print("✅ 一致：靜態選單與真 LIBERO 動態選單一字不差，本機 plan-only 對映可信。")
        return

    print("❌ 不一致，逐項差異：")
    for task_id in range(max(len(static), len(live))):
        s = static[task_id] if task_id < len(static) else None
        l = live[task_id] if task_id < len(live) else None
        if s != l:
            print(f"  task {task_id}: static={s!r}  live={l!r}")
    raise SystemExit("選單已偏差，請更新 LIBERO_OBJECT_TASKS。")


if __name__ == "__main__":
    main()
