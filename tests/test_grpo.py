import pytest

from grpo import group_metrics_by_target
from guess_env import RolloutMetrics


def metric(won: bool, turn_count: int) -> RolloutMetrics:
    return RolloutMetrics(
        won=won,
        turn_count=turn_count,
        out_of_range_count=0,
        repeated_count=0,
        direction_events=0,
        wrong_direction_count=0,
        reward_by_max_turns={5: 1.0 if won else 0.0},
    )


def test_group_metrics_by_target_preserves_submission_order():
    metrics = [metric(True, 1), metric(False, 5), metric(False, 3), metric(True, 2), metric(False, 4)]
    targets = [1, 1, 2, 2, 2]

    grouped = group_metrics_by_target(targets, metrics, (1, 2, 3))

    assert [m.turn_count for m in grouped[1]] == [1, 5]
    assert [m.turn_count for m in grouped[2]] == [3, 2, 4]
    assert grouped[3] == []


def test_group_metrics_by_target_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        group_metrics_by_target([1], [], (1,))
