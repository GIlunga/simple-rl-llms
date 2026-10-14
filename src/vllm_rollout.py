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

_VLLM_DTYPE = {torch.bfloat16: "bfloat16", torch.float16: "half", torch.float32: "float"}


def match_weight_names(vllm_names: list[str], hf_names: list[str]) -> dict[str, str]:
    """Map each HF parameter name to the unique vLLM parameter whose name it suffixes.

    Multimodal checkpoints nest the text model differently in HF (``model.layers...``) and vLLM
    (``language_model.model.layers...``), so we match on the trailing name rather than a fixed prefix.
    """
    mapping: dict[str, str] = {}
    for hf_name in hf_names:
        matches = [v for v in vllm_names if v == hf_name or v.endswith("." + hf_name)]
        if matches:
            mapping[hf_name] = min(matches, key=len)
    return mapping


def _get_vllm_parameter_names(worker) -> list[str]:
    return [name for name, _ in worker.model_runner.get_model().named_parameters()]


@dataclass
class _Rollout:
    env: GuessTheNumberEnv
    token_ids: list[int]
    loss_mask: list[bool]
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
        if params.vllm_kv_cache_bytes is not None:
            kwargs["kv_cache_memory_bytes"] = params.vllm_kv_cache_bytes

        self.llm = LLM(**kwargs)
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(
            temperature=params.temperature,
            max_tokens=params.max_tokens_per_turn,
            stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
        )
        self._name_map: dict[str, str] | None = None

    def sync_weights(self, policy_model) -> None:
        """Copy the trainer's parameters into the vLLM model (in-process, via ``collective_rpc``)."""
        if self._name_map is None:
            vllm_names = self.llm.collective_rpc(_get_vllm_parameter_names)[0]
            hf_names = [name for name, _ in policy_model.named_parameters()]
            self._name_map = match_weight_names(vllm_names, hf_names)

        weights = [
            (self._name_map[name], param.detach())
            for name, param in policy_model.named_parameters()
            if name in self._name_map
        ]
        self.llm.collective_rpc("reload_weights", args=(weights,))

    def generate_for_envs(
        self, envs: list[GuessTheNumberEnv], params: Parameters
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[RolloutMetrics]]:
        """Batched multi-turn rollouts, capped at ``rollout_batch_size`` concurrent envs.

        Envs are processed in chunks so the peak number of in-flight sequences (and therefore KV cache)
        stays bounded regardless of how many rollouts the step wants.
        """
        max_concurrency = params.rollout_batch_size

        token_seq_lst: list[torch.Tensor] = []
        output_mask_lst: list[torch.Tensor] = []
        metrics_lst: list[RolloutMetrics] = []
        for start in range(0, len(envs), max_concurrency):
            chunk_seqs, chunk_masks, chunk_metrics = self._generate_chunk(envs[start : start + max_concurrency], params)
            token_seq_lst.extend(chunk_seqs)
            output_mask_lst.extend(chunk_masks)
            metrics_lst.extend(chunk_metrics)

        return token_seq_lst, output_mask_lst, metrics_lst

    def _generate_chunk(
        self, envs: list[GuessTheNumberEnv], params: Parameters
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[RolloutMetrics]]:
        """One generation call per turn across the chunk's still-running envs."""
        from vllm import TokensPrompt

        rollouts = []
        for env in envs:
            prompt_ids = self._initial_prompt_ids(env, params)
            rollouts.append(_Rollout(env, prompt_ids, [False] * len(prompt_ids)))

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
                generated_ids = list(output.outputs[0].token_ids)
                rollout.token_ids.extend(generated_ids)
                rollout.loss_mask.extend([True] * len(generated_ids))

                text_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                result = rollout.env.step(text_response)

                if result.terminated or result.truncated:
                    rollout.done = True
                    continue

                observation_ids = self._observation_ids(result.observation)
                rollout.token_ids.extend(observation_ids)
                rollout.loss_mask.extend([False] * len(observation_ids))

        token_seq_lst = [torch.tensor(r.token_ids) for r in rollouts]
        output_mask_lst = [torch.tensor(r.loss_mask) for r in rollouts]
        metrics_lst = [r.env.rollout_metrics() for r in rollouts]
        return token_seq_lst, output_mask_lst, metrics_lst

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
