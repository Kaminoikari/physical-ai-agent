from agent.rollout_engine import RolloutOutcome


def test_rollout_outcome_holds_pc_success_and_optional_video():
    outcome = RolloutOutcome(pc_success=75.0)
    assert outcome.pc_success == 75.0
    assert outcome.video_path is None
