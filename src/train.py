from dataclasses import asdict, dataclass

import modal
import torch
import wandb

from build_helpers import MODEL_NAME, download_models
from grpo import train
from guess_env import EnvConfig


@dataclass(frozen=True)
class SamplingConfig:
    """Fully resolved rollout sampling knobs, shared by the vLLM and HF rollout paths."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float


# Qwen3.5's recommended sampling settings for text tasks, per thinking mode.
_QWEN_SAMPLING: dict[bool, SamplingConfig] = {
    False: SamplingConfig(
        temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0
    ),
    True: SamplingConfig(
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
    ),
}


@dataclass(frozen=True)
class Parameters:
    # Model/image settings
    model_name: str = MODEL_NAME
    use_thinking: bool = False
    # Sampling knobs: None = use Qwen's recommendation for the current thinking mode (see _QWEN_SAMPLING).
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    gpu: str = "L4"
    dtype: torch.dtype = torch.bfloat16
    timeout: int = 900  # seconds
    wandb_project: str = "SimpleGRPO"
    wandb_run_name: str = "GRPO no think"

    # vLLM rollout settings (collocated with training on the same GPU)
    use_vllm: bool = True
    # Max concurrent sequences per generate call. At gpu_memory_utilization=0.3 the KV cache holds
    # ~71 sequences of 2048 tokens, so 64 fits with margin; the current training/eval batch is 16.
    rollout_batch_size: int = 64
    vllm_gpu_memory_utilization: float = 0.3
    vllm_max_model_len: int = 2048
    vllm_kv_cache_bytes: int | None = None
    vllm_enforce_eager: bool = True
    vllm_enable_prefix_caching: bool = True
    vllm_language_model_only: bool = True
    vllm_seed: int = 0

    # GRPO settings
    num_iterations: int = 0
    num_steps: int = 2
    num_grpo_iterations: int = 1

    num_prompts_per_step: int = 4
    num_outputs_per_prompt: int = 4
    per_device_batch_size: int = 2

    kl_beta: float = 0.05
    importance_sampling_eps: float = 0.2
    max_grad_norm: float = 2.0

    # LR settings
    max_learning_rate: float = 5e-6
    min_learning_rate: float = 0.0
    warmup_ratio: float = 0.1
    decay_ratio: float = 0.1

    # Env settings
    min_number: int = 1
    max_number: int = 10
    max_turns: int = 5
    max_tokens_per_turn: int = 64

    # Reward settings
    reward_type: str = "binary"
    format_error_reward: float = 0.0
    invalid_action_reward: float = 0.0
    reward_breakdown_turns: tuple[int, ...] = (3, 4, 5)

    # Test set / eval settings
    run_test_set: bool = True
    eval_numbers: tuple[int, ...] = tuple(range(1, 11))
    eval_num_rollouts: int = 16
    # None = submit all (targets x rollouts) envs in one vLLM call; set an int to chunk.
    eval_rollout_batch_size: int | None = None
    eval_pass_at_k: tuple[int, ...] = (1, 2, 4, 8, 16)

    def sampling(self) -> SamplingConfig:
        """Resolve the rollout sampling knobs: explicit values win, unset ones fall back to Qwen's recommendation."""
        recommended = _QWEN_SAMPLING[self.use_thinking]
        return SamplingConfig(
            temperature=self.temperature if self.temperature is not None else recommended.temperature,
            top_p=self.top_p if self.top_p is not None else recommended.top_p,
            top_k=self.top_k if self.top_k is not None else recommended.top_k,
            min_p=self.min_p if self.min_p is not None else recommended.min_p,
            presence_penalty=(
                self.presence_penalty if self.presence_penalty is not None else recommended.presence_penalty
            ),
            repetition_penalty=(
                self.repetition_penalty if self.repetition_penalty is not None else recommended.repetition_penalty
            ),
        )


params = Parameters()

kernel_volume = modal.Volume.from_name("kernel-cache", create_if_missing=True)
tilelang_cache_volume = modal.Volume.from_name("tilelang-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "clang")
    .uv_sync()
    .run_function(download_models, secrets=[modal.Secret.from_name("huggingface-secret")])
    .add_local_python_source("guess_env", "grpo", "vllm_rollout")
)
app = modal.App("llm-rl-test", image=image)


def run_local() -> None:
    if params.max_turns not in params.reward_breakdown_turns:
        raise ValueError("params.max_turns must be one of params.reward_breakdown_turns")

    env_config = EnvConfig(
        min_number=params.min_number,
        max_number=params.max_number,
        max_turns=max(params.reward_breakdown_turns),
        reward_type=params.reward_type,
        format_error_reward=params.format_error_reward,
        invalid_action_reward=params.invalid_action_reward,
        reward_breakdown_turns=params.reward_breakdown_turns,
    )

    wandb_run = wandb.init(
        project=params.wandb_project,
        name=params.wandb_run_name,
        # Unset knobs are logged with the resolved Qwen recommendation, so a run records what it actually sampled with.
        config=asdict(params) | asdict(params.sampling()),
    )

    try:
        train(params, env_config, wandb_run)
    finally:
        wandb.finish()


@app.function(
    gpu=params.gpu,
    timeout=params.timeout,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    volumes={
        "/root/.triton": kernel_volume,
        "/root/.tilelang/cache": tilelang_cache_volume,
    },
)
def train_remote() -> None:
    run_local()


@app.local_entrypoint()
def main() -> None:
    train_remote.remote()


if __name__ == "__main__":
    run_local()
