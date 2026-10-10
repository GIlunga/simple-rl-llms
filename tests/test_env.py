import pytest

from guess_env import GuessTheNumberEnv

BASE = dict(
    min_number=1,
    max_number=10,
    max_turns=4,
    format_error_reward=-2.0,
    invalid_action_reward=-1.0,
    reward_breakdown_turns=(1, 2, 3, 4),
)
ONE = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
FORMAT = {1: -2.0, 2: -2.0, 3: -2.0, 4: -2.0}


@pytest.mark.parametrize(
    "reward_type, target, actions, expected_rewards, expected_won, expected_out_of_range, expected_repeated",
    [
        # happy path: number wrapped in surrounding text
        ("binary", 7, [r"I think the answer is \boxed{7}"], ONE, True, 0, 0),
        # multiple boxes: the LAST one wins
        ("binary", 7, [r"maybe \boxed{3}, but really \boxed{7}"], ONE, True, 0, 0),
        # multiple boxes where the last is wrong, then solve next turn
        (
            "binary",
            7,
            [r"\boxed{7} or maybe \boxed{2}", r"\boxed{7}"],
            {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0},
            True,
            0,
            0,
        ),
        # malformed: non-numeric inside the braces
        ("binary", 7, [r"the answer is \boxed{seven}"], FORMAT, False, 0, 0),
        # malformed: empty braces
        ("binary", 7, [r"\boxed{}"], FORMAT, False, 0, 0),
        # malformed: plain number with no \boxed{}
        ("binary", 7, ["the answer is 7"], FORMAT, False, 0, 0),
        # out-of-range guess in text, then solve
        (
            "binary",
            7,
            [r"way off: \boxed{42}", r"\boxed{7}"],
            {1: -1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            True,
            1,
            0,
        ),
        # repeated guess, then solve
        (
            "binary",
            7,
            [r"\boxed{1} please", r"again \boxed{1}", r"\boxed{7}"],
            {1: 0.0, 2: -1.0, 3: 1.0, 4: 1.0},
            True,
            0,
            1,
        ),
        # never solves -> truncated at max_turns, binary gives 0
        (
            "binary",
            7,
            [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}", r"\boxed{4}"],
            {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            False,
            0,
            0,
        ),
        # dense: partial credit on the last valid guess
        (
            "dense",
            7,
            [r"close: \boxed{6}"],
            {1: 1 - 1 / 9, 2: 1 - 1 / 9, 3: 1 - 1 / 9, 4: 1 - 1 / 9},
            False,
            0,
            0,
        ),
    ],
)
def test_env_failure_cases(
    reward_type, target, actions, expected_rewards, expected_won, expected_out_of_range, expected_repeated
):
    env = GuessTheNumberEnv(reward_type=reward_type, **BASE)
    env.reset(target=target)
    for action in actions:
        result = env.step(action)
        if result.terminated or result.truncated:
            break

    metrics = env.rollout_metrics()
    for budget, expected in expected_rewards.items():
        assert metrics.reward_by_max_turns[budget] == pytest.approx(expected)
    assert metrics.won == expected_won
    assert metrics.out_of_range_count == expected_out_of_range
    assert metrics.repeated_count == expected_repeated
