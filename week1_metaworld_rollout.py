"""Week 1 成功訊號：在 Meta-World 模擬裡看虛擬手臂執行一次 pick-place。

用 Meta-World 內建的 expert 腳本 policy 驅動 Sawyer 手臂完成 pick-place-v3，
逐幀渲染並存成影片。這只是 Week 1「看到手臂動起來」的驗證，與 SmolVLA 無關
（SmolVLA 驅動模擬要等 Week 2 對齊 action space 後微調）。
"""

import numpy as np
import imageio.v2 as imageio
import metaworld
import metaworld.policies as policies

TASK = "pick-place-v3"
CAMERA = "corner2"
MAX_STEPS = 200
OUT_PATH = "/Users/charles/LeRobot/week1_pickplace.mp4"


def main() -> None:
    mt1 = metaworld.MT1(TASK, seed=42)
    env = mt1.train_classes[TASK](render_mode="rgb_array", camera_name=CAMERA)
    env.set_task(mt1.train_tasks[0])
    # 拉遠相機，和 lerobot 的 metaworld 整合一致，畫面看得到整張桌子
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]

    expert = policies.SawyerPickPlaceV3Policy()

    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    succeeded = False
    for step in range(MAX_STEPS):
        action = np.clip(expert.get_action(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        # corner2 相機兩軸都是反的，翻正（與 lerobot 同處理）
        frames.append(np.flip(env.render(), (0, 1)))
        if int(info.get("success", 0)) == 1:
            succeeded = True
            print(f"✅ pick-place 成功！step={step}, reward={reward:.3f}")
            break

    # GIF：pillow 後端，免 codec，必成功
    gif_path = OUT_PATH.replace(".mp4", ".gif")
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"GIF 已存：{gif_path}")
    # mp4：用系統 ffmpeg（brew）明確指定 libx264，失敗不影響 GIF
    try:
        imageio.mimsave(OUT_PATH, frames, fps=30, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
        print(f"mp4 已存：{OUT_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"mp4 存檔略過（不影響）：{exc}")
    print(f"frames={len(frames)}, success={succeeded}")


if __name__ == "__main__":
    main()
