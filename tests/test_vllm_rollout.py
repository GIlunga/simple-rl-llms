from types import SimpleNamespace

from guess_env import EnvConfig, GuessTheNumberEnv
from vllm_rollout import VLLMRolloutEngine, checkpoint_name


def test_text_model_params_use_conditional_checkpoint_layout():
    assert checkpoint_name("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert checkpoint_name("model.layers.0.linear_attn.in_proj_qkv.weight") == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert checkpoint_name("model.layers.3.self_attn.q_proj.weight") == (
        "model.language_model.layers.3.self_attn.q_proj.weight"
    )


def test_lm_head_is_left_for_vllm_mapper():
    # The conditional model's mapper rewrites lm_head.* itself; it is not nested under model.
    assert checkpoint_name("lm_head.weight") == "lm_head.weight"


def test_unknown_prefix_passes_through():
    assert checkpoint_name("foo.bar.weight") == "foo.bar.weight"


class _FakeTokenizer:
    eos_token_id = 7
    im_end_id = 9

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return " ".join(message["content"] for message in messages)

    def __call__(self, text):
        return {"input_ids": [1, 2, 3]}

    def convert_tokens_to_ids(self, token):
        return self.im_end_id

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeLLM:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def generate(self, prompts, sampling_params, use_tqdm=False):
        logprobs = [-0.5 * (index + 1) for index in range(len(self.token_ids))]
        return [_FakeLoggedCompletion(self.token_ids, logprobs) for _ in prompts]


class _FakeLoggedCompletion:
    """RequestOutput whose completion carries one logprob entry per generated token."""

    def __init__(self, token_ids: list[int], logprobs: list[float]) -> None:
        entries = [{token_id: SimpleNamespace(logprob=logprob)} for token_id, logprob in zip(token_ids, logprobs)]
        self.outputs = [SimpleNamespace(token_ids=token_ids, logprobs=entries)]


def _env(target: int = 5) -> GuessTheNumberEnv:
    config = EnvConfig(
        min_number=1,
        max_number=10,
        max_turns=1,
        reward_type="binary",
        format_error_reward=0.0,
        invalid_action_reward=0.0,
        reward_breakdown_turns=(1,),
    )
    return GuessTheNumberEnv(config, target=target)


def test_generate_chunk_drops_trailing_endoftext_like_the_hf_path():
    # vLLM returns the stop token in token_ids; HF removes <|endoftext|> before concatenating the next turn.
    engine = object.__new__(VLLMRolloutEngine)
    engine.tokenizer = _FakeTokenizer()
    engine.llm = _FakeLLM([5, _FakeTokenizer.eos_token_id])
    engine.sampling_params = None

    token_ids, loss_mask, logprobs, _ = engine._generate_chunk([_env()], SimpleNamespace(use_thinking=False))

    assert token_ids[0].tolist() == [1, 2, 3, 5]
    assert loss_mask[0].sum().item() == 1
    # Prompt positions carry no logprob; the eos logprob is dropped together with its token.
    assert logprobs[0].tolist() == [0.0, 0.0, 0.0, -0.5]
