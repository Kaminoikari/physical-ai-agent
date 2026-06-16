# bench_rollout.py
"""Kaggle 實測：① in-process vs subprocess 的每 task 牆鐘時間；② 兩引擎成敗 parity。

用法（Kaggle，repo 根目錄，GPU）：
  python bench_rollout.py                 # SmolVLA，跑前幾個 task
  python bench_rollout.py --policy groot  # GR00T N1.5（需 Ampere+ GPU，見 spec 附錄 A）
只在 Kaggle/租用 GPU 跑；頂層僅 import 類別（lerobot 於引擎內 lazy import）。
"""

from __future__ import annotations

import argparse
import time

from agent.rollout_engine import (
    InProcessRolloutEngine,
    LerobotPolicyEnvBuilder,
    SubprocessRolloutEngine,
)

POLICY_SPECS = {
    "smolvla": ("smolvla", "HuggingFaceVLA/smolvla_libero"),
    "groot": ("groot", "nvidia/GR00T-N1.5-3B"),
}


def _time_engine(label, engine, task_ids):
    print(f"\n== {label} ==")
    results = {}
    for task_id in task_ids:
        start = time.perf_counter()
        outcome = engine.run(task_id, save_video=False, n_episodes=1)
        elapsed = time.perf_counter() - start
        results[task_id] = outcome.pc_success
        print(f"  task {task_id}: pc_success={outcome.pc_success:.1f}  {elapsed:.1f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=list(POLICY_SPECS), default="smolvla")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="0,1,2")
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()

    task_ids = [int(t) for t in args.tasks.split(",")]
    policy_type, policy_path = POLICY_SPECS[args.policy]

    in_proc = InProcessRolloutEngine(
        LerobotPolicyEnvBuilder(policy_type=policy_type, policy_path=policy_path,
                                suite=args.suite)
    )
    in_results = _time_engine(f"in-process [{args.policy}]", in_proc, task_ids)

    # baseline 僅對 SmolVLA 有意義（subprocess 走 lerobot-eval 預設 checkpoint）
    if args.policy == "smolvla":
        sub = SubprocessRolloutEngine(suite=args.suite, checkpoint=policy_path,
                                      device="cuda", output_root="/kaggle/working/bench")
        sub_results = _time_engine("subprocess [baseline]", sub, task_ids)

        print("\n== 成敗 parity（門檻 %.0f）==" % args.threshold)
        for task_id in task_ids:
            a = in_results[task_id] >= args.threshold
            b = sub_results[task_id] >= args.threshold
            mark = "✅" if a == b else "❌"
            print(f"  task {task_id}: in={a} sub={b} {mark}")


if __name__ == "__main__":
    main()
