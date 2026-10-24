from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from guess_env import GuessTheNumberEnv, RolloutMetrics

if TYPE_CHECKING:
    from transformers import AutoTokenizer

    from train import Parameters

# Run vLLM's engine core in this process (no separate worker), so generation and training
# share one process and one GPU. Must be set before vLLM reads its env vars.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# FlashInfer's JIT-compiled top-k/top-p sampler dominates startup profiling for this model
# (~90s of profile_run). The native sampler measured equally fast at runtime here, so disable it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_VLLM_DTYPE = {torch.bfloat16: "bfloat16", torch.float16: "half", torch.float32: "float"}


def checkpoint_name(hf_name: str) -> str:
    """Convert a text-only HF parameter name to vLLM's checkpoint layout.

    ``AutoModelForCausalLM`` resolves this checkpoint to the text-only ``Qwen3_5ForCausalLM``
    (``model.*``), while vLLM loads ``Qwen3_5ForConditionalGeneration``, whose checkpoint nests
    the text model under ``model.language_model.*``. Sending checkpoint names (rather than vLLM
    parameter names) lets vLLM apply its own HF->vLLM mapping and packed/fused projections.
    """
    if hf_name.startswith("model."):
        return "model.language_model." + hf_name[len("model.") :]
    return hf_name


@dataclass
class _Rollout:
    env: GuessTheNumberEnv
    token_ids: list[int]
    loss_mask: list[bool]
    token_logprobs: list[float]  # vLLM's logprob for each generated token, 0.0 elsewhere
    done: bool = False


class VLLMRolloutEngine:
    """In-process vLLM engine for batched multi-turn rollout generation (collocated with training)."""

    def __init__(self, model_name: str, tokenizer: AutoTokenizer, params: Parameters) -> None:
        from vllm import LLM, SamplingParams

        kwargs = {
            "model": model_name,
            "dtype": _VLLM_DTYPE.get(params.dtype, "bfloat16"),
            "max_model_len": params.vllm_max_model_len,
            "gpu_memory_utilization": params.vllm_gpu_memory_utilization,
            "enable_prefix_caching": params.vllm_enable_prefix_caching,
            "enforce_eager": params.vllm_enforce_eager,
            "seed": params.vllm_seed,
        }
        if params.vllm_language_model_only:
            kwargs["language_model_only"] = True
        if params.vllm_kv_cache_bytes is not None:
            kwargs["kv_cache_memory_bytes"] = params.vllm_kv_cache_bytes

        self.llm = LLM(**kwargs)
        self.tokenizer = tokenizer
        sampling = params.sampling()
        self.sampling_params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            presence_penalty=sampling.presence_penalty,
            repetition_penalty=sampling.repetition_penalty,
            max_tokens=params.max_tokens_per_turn,
            stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
            # Only the sampled token's logprob ("raw" mode by default: pre-temperature), for the sync check
            # in grpo.logprob_gap_metrics.
            logprobs=0,
        )

    def sync_weights(self, policy_model) -> None:
        """Copy the trainer's parameters into the vLLM model (in-process, via ``collective_rpc``).

        Weights are sent in checkpoint layout so vLLM's ``load_weights`` applies its HF->vLLM
        mapping and fuses/stacks packed projections (qkv, gate_up, GDN in_proj) itself.
        """
        weights = [(checkpoint_name(name), param.detach()) for name, param in policy_model.named_parameters()]
        self.llm.collective_rpc("reload_weights", args=(weights,))

    def generate_for_envs(
        self,
        envs: list[GuessTheNumberEnv],
        params: Parameters,
        max_concurrency: int | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[RolloutMetrics]]:
        """Batched multi-turn rollouts, capped at ``max_concurrency`` concurrent envs.

        ``None`` (or a non-positive value) means no cap: every env is submitted in one chunk.
        Training passes ``rollout_batch_size``; eval passes ``eval_rollout_batch_size`` (``None``
        by default) so all targets run together. Sequences, masks, per-token vLLM logprobs, and
        metrics are returned in the input env order.
        """
        if not envs:
            return [], [], [], []

        if max_concurrency is None or max_concurrency <= 0:
            max_concurrency = len(envs)

        token_seq_lst: list[torch.Tensor] = []
        output_mask_lst: list[torch.Tensor] = []
        logprob_lst: list[torch.Tensor] = []
        metrics_lst: list[RolloutMetrics] = []
        for start in range(0, len(envs), max_concurrency):
            chunk_seqs, chunk_masks, chunk_logprobs, chunk_metrics = self._generate_chunk(
                envs[start : start + max_concurrency], params
            )
            token_seq_lst.extend(chunk_seqs)
            output_mask_lst.extend(chunk_masks)
            logprob_lst.extend(chunk_logprobs)
            metrics_lst.extend(chunk_metrics)

        return token_seq_lst, output_mask_lst, logprob_lst, metrics_lst

    def _generate_chunk(
        self, envs: list[GuessTheNumberEnv], params: Parameters
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[RolloutMetrics]]:
        """One generation call per turn across the chunk's still-running envs."""
        from vllm import TokensPrompt

        rollouts = []
        for env in envs:
            prompt_ids = self._initial_prompt_ids(env, params)
            rollouts.append(_Rollout(env, prompt_ids, [False] * len(prompt_ids), [0.0] * len(prompt_ids)))

        while True:
            active = [r for r in rollouts if not r.done]
            if not active:
                break

            outputs = self.llm.generate(
                [TokensPrompt(prompt_token_ids=r.token_ids) for r in active],
                self.sampling_params,
                use_tqdm=False,
            )

            for rollout, output in zip(active, outputs, strict=True):
                completion = output.outputs[0]
                generated_ids = list(completion.token_ids)
                if len(completion.logprobs) != len(generated_ids):
                    raise ValueError(
                        f"vLLM returned {len(completion.logprobs)} logprob entries for "
                        f"{len(generated_ids)} generated tokens"
                    )
                generated_logprobs = [
                    float(entry[token_id].logprob) for token_id, entry in zip(generated_ids, completion.logprobs)
                ]
                # Mirror the HF path (grpo.generate_single_rollout): <|im_end|> is the turn separator, so a
                # trailing <|endoftext|> must not be carried into the next turn's prompt.
                if generated_ids and generated_ids[-1] == self.tokenizer.eos_token_id:
                    generated_ids.pop()
                    generated_logprobs.pop()
                rollout.token_ids.extend(generated_ids)
                rollout.loss_mask.extend([True] * len(generated_ids))
                rollout.token_logprobs.extend(generated_logprobs)

                text_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                result = rollout.env.step(text_response)

                if result.terminated or result.truncated:
                    rollout.done = True
                    continue

                observation_ids = self._observation_ids(result.observation)
                rollout.token_ids.extend(observation_ids)
                rollout.loss_mask.extend([False] * len(observation_ids))
                rollout.token_logprobs.extend([0.0] * len(observation_ids))

        token_seq_lst = [torch.tensor(r.token_ids) for r in rollouts]
        output_mask_lst = [torch.tensor(r.loss_mask) for r in rollouts]
        logprob_seq_lst = [torch.tensor(r.token_logprobs, dtype=torch.float32) for r in rollouts]
        metrics_lst = [r.env.rollout_metrics() for r in rollouts]
        return token_seq_lst, output_mask_lst, logprob_seq_lst, metrics_lst

    def _initial_prompt_ids(self, env: GuessTheNumberEnv, params: Parameters) -> list[int]:
        messages = [
            {"content": env.system_prompt, "role": "system"},
            {"content": env.initial_message, "role": "user"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=params.use_thinking, add_generation_prompt=True
        )
        return self.tokenizer(text)["input_ids"]

    def _observation_ids(self, observation: str) -> list[int]:
        # Mirrors generate_single_rollout: the per-turn observation template does not pass enable_thinking.
        text = self.tokenizer.apply_chat_template(
            [{"content": observation, "role": "user"}], tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text)["input_ids"]
