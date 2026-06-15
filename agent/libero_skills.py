"""L1+L2（真實版）：LiberoSkillInterface — task-level 技能，驅動真 LIBERO + SmolVLA。

與 mock 的 SkillInterface 同樣被 agent.py 呼叫，但技能粒度是「整段 task 的 rollout」：
- available_tasks(): 列出該 suite 每個任務的自然語言指令（= agent 的技能選單）
- execute(task):    跑一次 LIBERO rollout，回傳 LIBERO 的 ground-truth success

⚠️ 只能在 Linux + GPU（Kaggle）跑，無法在 Mac 本機執行。所有重依賴皆 lazy import，
   讓本機 import 此模組不致失敗（但 execute/available_tasks 實際呼叫需在 Kaggle）。

v1 工程取捨：execute 以 subprocess 呼叫已驗證的 `lerobot-eval`（重用整條官方管線、
最穩），代價是每次重載 policy、較慢。日後可改為「載入 policy 一次 + 直接呼叫
lerobot 的 eval_policy()」加速（見 docs spec 的風險段）。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

from agent.schemas import SkillResult

DEFAULT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"


class LiberoSkillInterface:
    def __init__(
        self,
        suite: str = "libero_object",
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        n_episodes: int = 1,
        success_threshold: float = 50.0,  # pc_success(%) 達標即視為成功
    ) -> None:
        self.suite = suite
        self.checkpoint = checkpoint
        self.device = device
        self.n_episodes = n_episodes
        self.success_threshold = success_threshold
        self._tasks = self._load_task_list()

    def _load_task_list(self) -> list[tuple[int, str]]:
        from libero.libero import benchmark  # lazy：只在 Kaggle 需要

        suite_cls = benchmark.get_benchmark_dict()[self.suite]
        task_suite = suite_cls()
        return [(i, task_suite.get_task(i).language) for i in range(task_suite.n_tasks)]

    def available_tasks(self) -> list[tuple[int, str]]:
        return list(self._tasks)

    def _resolve_task_id(self, task: str) -> int:
        """agent 給的 arg 可能是 task id（推薦）或語言字串；兩種都解析。"""
        task = task.strip()
        if task.isdigit():
            return int(task)
        # 退而求其次：用語言字串做寬鬆比對
        lowered = task.lower()
        for task_id, language in self._tasks:
            if lowered in language.lower() or language.lower() in lowered:
                return task_id
        raise ValueError(f"無法對應到任何 LIBERO 任務：{task!r}")

    def execute(self, task: str) -> SkillResult:
        task_id = self._resolve_task_id(task)
        language = self._tasks[task_id][1]
        success = self._run_eval(task_id)
        return SkillResult(
            ok=success,
            detail=f"execute(task {task_id}: {language}) -> {'成功' if success else '失敗'}",
        )

    def _run_eval(self, task_id: int) -> bool:
        out_dir = tempfile.mkdtemp(prefix="libero_exec_")
        cmd = [
            "lerobot-eval",
            f"--policy.path={self.checkpoint}",
            f"--policy.device={self.device}",
            "--env.type=libero",
            f"--env.task={self.suite}",
            f"--env.task_ids=[{task_id}]",
            "--eval.batch_size=1",
            f"--eval.n_episodes={self.n_episodes}",
            "--env.max_parallel_tasks=1",
            f"--output_dir={out_dir}",
        ]
        env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        info_path = os.path.join(out_dir, "eval_info.json")
        if proc.returncode != 0 or not os.path.exists(info_path):
            raise RuntimeError(
                f"lerobot-eval 失敗（task {task_id}）：\n{proc.stderr[-2000:]}"
            )
        with open(info_path) as f:
            info = json.load(f)
        return float(info["overall"]["pc_success"]) >= self.success_threshold

    # 與 mock 對稱：語意查詢（這裡用任務清單回答「有哪些任務」）
    def query(self, question: str, mode: str = "semantic") -> str:
        return "可用任務：" + "；".join(f"{i}={lang}" for i, lang in self._tasks)
