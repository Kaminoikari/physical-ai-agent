"""即時互動檢視 Meta-World pick-place（MuJoCo 原生 3D viewer）。

⚠️ macOS 必須用 mjpython 啟動（GUI 要跑在主執行緒）：
    .venv/bin/mjpython week1_metaworld_viewer.py

操作：滑鼠左鍵拖曳=繞鏡頭、右鍵拖曳=平移、滾輪=縮放、空白鍵=暫停、關視窗=結束。
expert 腳本 policy 即時驅動 Sawyer 手臂，成功後自動重置、無限循環。
"""

import time

import numpy as np
import mujoco
import mujoco.viewer
import metaworld
import metaworld.policies as policies

TASK = "pick-place-v3"


def main() -> None:
    mt1 = metaworld.MT1(TASK, seed=42)
    env = mt1.train_classes[TASK](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])

    expert = policies.SawyerPickPlaceV3Policy()
    obs, _ = env.reset(seed=42)

    # 與 env 共用同一份 mujoco model/data，viewer.sync() 才會反映每次 step
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action = np.clip(expert.get_action(obs), -1.0, 1.0)
            obs, _, _, _, info = env.step(action)
            viewer.sync()
            time.sleep(env.dt)  # 貼近真實時間速度
            if int(info.get("success", 0)) == 1:
                time.sleep(0.6)  # 成功後停一下讓你看清楚
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
