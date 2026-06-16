from agent.rollout_engine import RolloutOutcome, InProcessRolloutEngine, PolicyBundle


def test_rollout_outcome_holds_pc_success_and_optional_video():
    outcome = RolloutOutcome(pc_success=75.0)
    assert outcome.pc_success == 75.0
    assert outcome.video_path is None


class FakeBuilder:
    policy_type = "smolvla"
    policy_path = "fake/checkpoint"

    def __init__(self):
        self.build_policy_calls = 0
        self.build_env_calls: list[int] = []

    def build_policy(self) -> PolicyBundle:
        self.build_policy_calls += 1
        return PolicyBundle("POLICY", "envpre", "envpost", "pre", "post")

    def build_env(self, task_id: int):
        self.build_env_calls.append(task_id)
        return f"ENV{task_id}"


class FakeEval:
    """記錄被呼叫的引數，回固定 info（仿 eval_policy 回傳結構）。"""

    def __init__(self, pc_success=80.0):
        self.calls: list[dict] = []
        self._pc = pc_success

    def __call__(self, env, policy, env_pre, env_post, pre, post, *, n_episodes,
                 max_episodes_rendered, videos_dir):
        self.calls.append(
            {"env": env, "policy": policy, "n_episodes": n_episodes,
             "max_episodes_rendered": max_episodes_rendered, "videos_dir": videos_dir}
        )
        return {"aggregated": {"pc_success": self._pc}, "video_paths": []}


def test_inprocess_builds_policy_once_across_runs():
    builder = FakeBuilder()
    engine = InProcessRolloutEngine(builder, eval_fn=FakeEval())
    for _ in range(3):
        engine.run(0, save_video=False, n_episodes=1)
    assert builder.build_policy_calls == 1


def test_inprocess_caches_env_per_task_id():
    builder = FakeBuilder()
    engine = InProcessRolloutEngine(builder, eval_fn=FakeEval())
    engine.run(0, save_video=False, n_episodes=1)
    engine.run(0, save_video=False, n_episodes=1)
    engine.run(2, save_video=False, n_episodes=1)
    assert builder.build_env_calls == [0, 2]


def test_inprocess_save_video_flag_controls_render_and_videos_dir():
    fake_eval = FakeEval()
    engine = InProcessRolloutEngine(FakeBuilder(), eval_fn=fake_eval, videos_dir="/tmp/v",
                                    max_render_episodes=1)
    engine.run(0, save_video=False, n_episodes=1)
    engine.run(0, save_video=True, n_episodes=1)
    assert fake_eval.calls[0]["max_episodes_rendered"] == 0
    assert fake_eval.calls[0]["videos_dir"] is None
    assert fake_eval.calls[1]["max_episodes_rendered"] == 1
    assert fake_eval.calls[1]["videos_dir"] == "/tmp/v"


def test_inprocess_returns_pc_success_from_eval_info():
    engine = InProcessRolloutEngine(FakeBuilder(), eval_fn=FakeEval(pc_success=42.0))
    outcome = engine.run(0, save_video=False, n_episodes=1)
    assert outcome.pc_success == 42.0
