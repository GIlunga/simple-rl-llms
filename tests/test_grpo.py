import pytest
import torch

from grpo import (
    _PresencePenaltyLogitsProcessor,
    group_metrics_by_target,
    logprob_gap_metrics,
    sampling_logits_processor,
)
from guess_env import RolloutMetrics
from train import Parameters, SamplingConfig


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


def _sampling(**overrides) -> SamplingConfig:
    values = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0)
    values.update(overrides)
    return SamplingConfig(**values)


def test_sampling_defaults_follow_qwen_recommendations():
    default = Parameters().sampling()
    assert (default.temperature, default.top_p, default.top_k, default.min_p) == (1.0, 1.0, 20, 0.0)
    assert (default.presence_penalty, default.repetition_penalty) == (2.0, 1.0)

    thinking = Parameters(use_thinking=True).sampling()
    assert (thinking.top_p, thinking.presence_penalty) == (0.95, 1.5)


def test_explicit_sampling_params_override_qwen_recommendations():
    sampling = Parameters(temperature=0.4, top_k=50, presence_penalty=0.0).sampling()
    assert (sampling.temperature, sampling.top_k, sampling.presence_penalty) == (0.4, 50, 0.0)
    assert sampling.top_p == 1.0  # still the recommendation


def test_sampling_chain_matches_vllm_order():
    chain = sampling_logits_processor(
        _sampling(temperature=0.5, top_p=0.9, top_k=5, min_p=0.1, presence_penalty=2.0, repetition_penalty=1.2),
        prompt_len=1,
    )
    assert [type(processor).__name__ for processor in chain] == [
        "RepetitionPenaltyLogitsProcessor",
        "_PresencePenaltyLogitsProcessor",
        "TemperatureLogitsWarper",
        "MinPLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
    ]


def test_presence_penalty_only_penalizes_tokens_generated_in_this_call():
    processor = _PresencePenaltyLogitsProcessor(penalty=2.0, prompt_len=2)

    scores = processor(torch.tensor([[1, 3, 4, 3]]), torch.zeros(1, 6))

    # Prompt tokens stay untouched even when they re-appear in the generated part.
    assert scores.tolist() == [[0.0, 0.0, 0.0, -2.0, -2.0, 0.0]]


def test_penalty_is_applied_before_top_k():
    chain = sampling_logits_processor(_sampling(top_k=1, presence_penalty=10.0), prompt_len=1)

    scores = torch.tensor([[-1.0, 1.0, 5.0]])
    input_ids = torch.tensor([[0, 2]])  # token 2 (the raw argmax) was already generated this turn
    for processor in chain:
        scores = processor(input_ids, scores)

    assert scores[0, 2].item() == float("-inf")
    assert scores[0, 1].item() == 1.0


def test_logprob_gap_metrics_lines_up_vllm_logprobs_with_the_shifted_hf_view():
    # Prompt token 0 has no logprob; generated tokens sit at positions 1..2 (row 0) and 1 (row 1).
    vllm_logprobs = torch.tensor([[0.0, -0.5, -1.0, 0.0], [0.0, -2.0, 0.0, 0.0]])
    # hf_logprobs[:, i] scores token_seqs[:, i + 1] -> compare against vllm_logprobs[:, i + 1].
    hf_logprobs = torch.tensor([[-0.6, -0.8, 0.0], [-2.2, 0.0, 0.0]])
    loss_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    metrics = logprob_gap_metrics(vllm_logprobs, hf_logprobs, loss_mask)

    # Gaps are -0.1, +0.2 (row 0) and -0.2 (row 1).
    assert metrics["Logprob gap/mean signed"] == pytest.approx(-1 / 30)
    assert metrics["Logprob gap/mean abs"] == pytest.approx(0.5 / 3)
    assert metrics["Logprob gap/max abs"] == pytest.approx(0.2)
