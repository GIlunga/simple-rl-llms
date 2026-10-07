from dataclasses import asdict, dataclass

import modal
import torch
import wandb

from build_helpers import MODEL_NAME, download_models
from grpo import train


@dataclass(frozen=True)
class Parameters:
    # Model/image settings
    model_name: str = MODEL_NAME
    use_thinking: bool = False
    temperature: float = 0.4
    gpu: str = "L4"
    dtype: torch.dtype = torch.bfloat16
    timeout: int = 900  # seconds
    wandb_project: str = "SimpleGRPO"
    wandb_run_name: str = "GRPO no think"

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
    eval_pass_at_k: tuple[int, ...] = (1, 2, 4, 8, 16)


params = Parameters()

kernel_volume = modal.Volume.from_name("kernel-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "clang")
    .uv_sync()
    .run_function(download_models, secrets=[modal.Secret.from_name("huggingface-secret")])
    .add_local_python_source("env", "grpo")
)
app = modal.App("llm-rl-test", image=image)


def run_local() -> None:
    wandb_run = wandb.init(project=params.wandb_project, name=params.wandb_run_name, config=asdict(params))

    try:
        train(params, wandb_run)
    finally:
        wandb.finish()


@app.function(
    gpu=params.gpu,
    timeout=params.timeout,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    volumes={"/root/.triton": kernel_volume},
)
def train_remote() -> None:
    run_local()


@app.local_entrypoint()
def main() -> None:
    train_remote.remote()


if __name__ == "__main__":
    run_local()
