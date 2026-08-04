import logging
import math
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass

import modal
import torch
import torch.nn.functional as F
import wandb
from gem.envs.game_env.guess_the_number import GuessTheNumberEnv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable annoying prints
warnings.filterwarnings("ignore", message=".*tl.make_block_ptr is deprecated.*")
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Parameters:
    # Modal settings
    model_name = "Qwen/Qwen3.5-0.8B"
    gpu = "L4"
    dtype = torch.bfloat16
    timeout = 400  # seconds
    wandb_project = "Simple RL for LLMs"
    wandb_run_name = None

    # GRPO settings
    num_iterations = 2  # Outer loop, sync reference model
    num_steps = 2  # Mid loop, sample prompts + outputs to train grpo_iterations times
    num_grpo_iterations = 1  # Inner loop, GRPO update on step dataset (replay)

    num_prompts_per_step = 2
    num_outputs_per_prompt = 2
    per_device_batch_size = 4

    kl_beta = 0.05
    importance_sampling_eps = 0.2
    max_grad_norm = 2

    max_learning_rate = 5e-6
    min_learning_rate = 0
    warmup_ratio = 0.05
    decay_ratio = 0.1

    # Env settings
    min_number = 1
    max_number = 10
    max_turns = 5
    max_tokens_per_turn = 32


params = Parameters()


# Modal image definition
def download_models():
    # Helper for Modal image caching
    from huggingface_hub import snapshot_download

    snapshot_download(params.model_name)


kernel_volume = modal.Volume.from_name("kernel-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "clang")
    .uv_sync()
    .run_function(download_models, secrets=[modal.Secret.from_name("huggingface-secret")])
)
app = modal.App("llm-rl-test", image=image)


# Visualisation
console = Console(force_terminal=True)


def print_masked_sequence(
    sequences: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    *,
    won: bool,
    turn_count: int,
    reward: float,
) -> None:
    """Print LLM generated text in green, other text muted"""
    text = Text()
    for tok, m in zip(sequences.tolist(), mask.tolist(), strict=True):
        decoded = tokenizer.decode([tok]).replace("\n", "\\n")
        text.append(decoded, style="bold green" if m else "dim")

    won_style = "green" if won else "red"
    reward_style = "green" if reward >= 1.0 else ("yellow" if reward > 0 else "red")
    title = (
        f"[white]won=[/][{won_style}]{won}[/]  "
        f"[white]turns={turn_count}[/]  "
        f"[white]reward=[/][{reward_style}]{round(reward, 2)}[/]"
    )
    console.print(Panel(text, title=title, border_style="bright_black"))


def print_rollouts(
    token_seqs: torch.Tensor,
    loss_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    rewards: torch.Tensor,
    won_lst: list[bool],
    turn_count_lst: list[int],
    tokenizer,
    num_completions_per_prompt: int,
) -> None:
    num_prompts = len(won_lst) // num_completions_per_prompt
    for prompt_idx in range(num_prompts):
        console.rule(f"[bold]Prompt {prompt_idx + 1}[/]")
        start = prompt_idx * num_completions_per_prompt
        end = start + num_completions_per_prompt
        for i in range(start, end):
            real_len = attn_mask[i].sum().item()
            print_masked_sequence(
                token_seqs[i, :real_len],
                loss_mask[i, :real_len],
                tokenizer,
                won=won_lst[i],
                turn_count=turn_count_lst[i],
                reward=rewards[i].item(),
            )


# Actual GRPO algo
def generate_single_rollout(env, model, tokenizer, max_tokens_per_turn):
    message_list = [
        {
            "content": "You are playing Guess The Number with the user. You have to guess the number between 1 and 10 "
            "(inclusive) within 5 turns. As you enter your guess, the user will provide you with hints such as the "
            "target number is 'higher' or 'lower'. When answering, only the number that is wrapped inside \\boxed{} "
            "will be considered as your guess, for example, \\boxed{10}. Follow that exact format for your final "
            "answer.",
            "role": "system",
        },
        {"content": "Enter your first guess to start the game!", "role": "user"},
    ]

    model.eval()

    terminated = False
    truncated = False

    # Thinking = False adds a thinking section and closes it immediately
    inputs_text = tokenizer.apply_chat_template(
        message_list, tokenize=False, enable_thinking=False, add_generation_prompt=True
    )

    inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    prev_len = inputs["input_ids"].shape[1]
    output_mask = [False] * prev_len  # Mask out system prompt + special tokens
    end_of_text_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    im_end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Iterate multi-step env
    while True:
        with torch.inference_mode():
            output_dict = model.generate(
                **inputs,
                max_new_tokens=max_tokens_per_turn,
                temperature=1.0,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, im_end_token],
            )

        # Strip end of text
        if output_dict.sequences[0][-1] == end_of_text_token_id:
            output_dict.sequences = output_dict.sequences[:, :-1]

        # Env step
        text_response = tokenizer.decode(output_dict.sequences[0][prev_len:], skip_special_tokens=True)
        observation, reward, terminated, truncated, _ = env.step(text_response)

        # Update mask with model response
        output_mask += [True] * (output_dict.sequences.shape[1] - prev_len)

        # Add new text
        observation_msg = {"role": "user", "content": observation}
        message_list.extend([{"role": "assistant", "content": text_response}, observation_msg])

        if terminated or truncated:
            break

        new_inputs = tokenizer.apply_chat_template([observation_msg], tokenize=False, add_generation_prompt=True)

        new_inputs = tokenizer(new_inputs, return_tensors="pt").to(model.device)

        inputs["input_ids"] = torch.cat([output_dict.sequences, new_inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # Update mask to ignore observation and assistant start token
        output_mask += [False] * (inputs["input_ids"].shape[1] - output_dict.sequences.shape[1])
        prev_len = inputs["input_ids"].shape[1]

    return (
        output_dict.sequences.detach().cpu(),
        torch.tensor(output_mask),
        reward,
        reward == 1.0,
        env.turn_count,
    )


def get_rollouts(
    base_env,
    policy_model,
    reference_model,
    tokenizer,
    max_tokens_per_turn,
    num_prompts_per_step,
    num_completions_per_prompt,
):
    """Simple sequential multiple rollout generation for multiple prompts. No batching"""
    token_seq_lst = []
    output_mask_lst = []
    reward_lst = []
    advantage_lst = []
    won_lst = []
    turn_count_lst = []

    for _ in range(num_prompts_per_step):
        base_env.reset()
        # Required to maintain same target for guess the number (env copies share same target number)
        env_copies = [deepcopy(base_env) for _ in range(num_completions_per_prompt)]
        group_rewards = []
        for env in env_copies:
            token_seq, output_mask, reward, won, turn_count = generate_single_rollout(
                env, policy_model, tokenizer, max_tokens_per_turn
            )

            token_seq_lst.append(token_seq.squeeze())
            output_mask_lst.append(output_mask)
            group_rewards.append(reward)
            won_lst.append(won)
            turn_count_lst.append(turn_count)

        group_rewards = torch.tensor(group_rewards)

        # GRPO advantage normalization
        advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)

        reward_lst.append(group_rewards)
        advantage_lst.append(advantages)

    # shape: B x T
    token_seqs = torch.nn.utils.rnn.pad_sequence(token_seq_lst, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = token_seqs != tokenizer.pad_token_id

    loss_mask = torch.nn.utils.rnn.pad_sequence(output_mask_lst, batch_first=True, padding_value=False)
    loss_mask = (loss_mask & attn_mask).float()

    # Get logprobs for importance sampling + KL
    gen_policy_logprobs = get_logprobs_from_rollouts(policy_model, token_seqs, loss_mask, attn_mask)
    ref_model_logprobs = get_logprobs_from_rollouts(reference_model, token_seqs, loss_mask, attn_mask)

    # Compute dataset metrics
    total_size, _ = token_seqs.shape
    all_rewards = torch.cat(reward_lst)
    reward_matrix = all_rewards.view(-1, num_completions_per_prompt)
    rollout_lens = loss_mask[:, 1:].sum(dim=1)
    dataset_metrics = {
        "Rollout Len/avg": rollout_lens.mean().item(),
        "Rollout Len/max": rollout_lens.max().item(),
        "Rollout Len/min": rollout_lens.min().item(),
        "Rewards/avg": all_rewards.mean().item(),
        "Rewards/std": all_rewards.std().item(),
        "Rewards/max": all_rewards.max().item(),
        "Rewards/min": all_rewards.min().item(),
        "Rewards/zero_rate": (all_rewards <= 0).sum().item() / total_size,
        "Rewards/one_rate": (all_rewards == 1).sum().item() / total_size,
        "Rewards/zero_group_rate": (reward_matrix <= 0).all(dim=1).float().mean().item(),
        "Rewards/one_group_rate": (reward_matrix == 1).all(dim=1).float().mean().item(),
    }

    # Print rollouts to stdout
    print_rollouts(
        token_seqs,
        loss_mask,
        attn_mask,
        all_rewards,
        won_lst,
        turn_count_lst,
        tokenizer,
        num_completions_per_prompt,
    )

    return (
        token_seqs,
        loss_mask,
        attn_mask,
        torch.cat(advantage_lst),
        dataset_metrics,
        gen_policy_logprobs,
        ref_model_logprobs,
    )


def get_logprobs_from_rollouts(policy_model, token_seqs, loss_mask, attn_mask):
    total_size, _ = token_seqs.shape
    num_batches = (total_size + params.per_device_batch_size - 1) // params.per_device_batch_size
    old_policy_logprobs_list = []
    with torch.inference_mode():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * params.per_device_batch_size
            end_idx = (batch_idx + 1) * params.per_device_batch_size

            batch_inputs = token_seqs[start_idx:end_idx].cuda()
            batch_attn_mask = attn_mask[start_idx:end_idx].cuda()
            batch_loss_mask = loss_mask[start_idx:end_idx].cuda()
            targets = batch_inputs[:, 1:]
            shifted_loss_mask = batch_loss_mask[:, 1:]

            outputs = policy_model.forward(batch_inputs, attention_mask=batch_attn_mask)
            logprobs = get_logprobs_from_logits(outputs, targets) * shifted_loss_mask
            old_policy_logprobs_list.append(logprobs.cpu())

    return torch.cat(old_policy_logprobs_list, dim=0)


def get_logprobs_from_logits(model_output, targets):
    # Shape: batch x seq_len x vocab
    shifted_logits = model_output.logits[:, :-1, :]
    flat_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))
    flat_targets = targets.reshape(-1)

    # Cross entropy outputs the negative log likelihood of the sequence
    return -F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(
        shifted_logits.shape[0], shifted_logits.shape[1]
    )


def create_wsd_scheduler(optimizer, total_steps):
    num_warmup_steps = int(params.warmup_ratio * total_steps)
    stable_ratio = 1 - params.warmup_ratio - params.decay_ratio
    num_stable_steps = int(stable_ratio * total_steps)
    num_decay_steps = total_steps - num_warmup_steps - num_stable_steps

    min_lr_ratio = params.min_learning_rate / params.max_learning_rate

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps

        if current_step < (num_warmup_steps + num_stable_steps):
            return 1.0

        decay_progress = (current_step - num_warmup_steps - num_stable_steps) / num_decay_steps

        decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * decay_factor

    return LambdaLR(optimizer, lr_lambda)


def train_grpo(wandb_run):
    policy_model = AutoModelForCausalLM.from_pretrained(params.model_name, device_map="cuda", dtype=params.dtype)
    reference_model = AutoModelForCausalLM.from_pretrained(
        params.model_name, device_map="cuda", dtype=params.dtype
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    base_env = GuessTheNumberEnv(min_number=params.min_number, max_number=params.max_number, max_turns=params.max_turns)

    optimizer = torch.optim.AdamW(policy_model.parameters(), params.max_learning_rate)
    wsd_scheduler = create_wsd_scheduler(
        optimizer, total_steps=params.num_iterations * params.num_steps * params.num_grpo_iterations
    )

    for iteration in range(1, params.num_iterations + 1):
        if iteration > 1:
            print("Syncing reference model")
            reference_model.load_state_dict(policy_model.state_dict())

        for step in range(1, params.num_steps + 1):
            # Generate batch for this step (num_prompts * num_outputs_per_prompt)
            (
                token_seqs,
                loss_mask,
                attn_mask,
                advantages,
                dataset_metrics,
                old_policy_logprobs,
                ref_model_logprobs,
            ) = get_rollouts(
                base_env,
                policy_model,
                reference_model,
                tokenizer,
                params.max_tokens_per_turn,
                params.num_prompts_per_step,
                params.num_outputs_per_prompt,
            )

            total_size, _ = token_seqs.shape

            # Inner loop - train multiple times on the same batch, each iteration is 1 optimization step
            # Split batch and accumulate gradients according to per_device_batch_size
            policy_model.train()
            num_batches = (total_size + params.per_device_batch_size - 1) // params.per_device_batch_size
            for grpo_iteration in range(1, params.num_grpo_iterations + 1):
                acc_loss = 0.0
                acc_kl = 0.0
                global_step = step * params.num_grpo_iterations

                optimizer.zero_grad()
                for batch_idx in range(num_batches):
                    # Get batch from dataset
                    start_idx = batch_idx * params.per_device_batch_size
                    end_idx = (batch_idx + 1) * params.per_device_batch_size

                    batch_inputs = token_seqs[start_idx:end_idx].cuda()
                    batch_attn_mask = attn_mask[start_idx:end_idx].cuda()
                    batch_loss_mask = loss_mask[start_idx:end_idx].cuda()
                    batch_advantages = advantages[start_idx:end_idx].cuda()
                    batch_old_logprobs = old_policy_logprobs[start_idx:end_idx].cuda()
                    batch_ref_logprobs = ref_model_logprobs[start_idx:end_idx].cuda()
                    targets = batch_inputs[:, 1:]

                    # Get current policy model logprobs
                    shifted_loss_mask = batch_loss_mask[:, 1:]  # Shift mask by 1 to match logprobs
                    policy_model_output = policy_model.forward(batch_inputs, attention_mask=batch_attn_mask)
                    policy_model_logprobs = get_logprobs_from_logits(policy_model_output, targets) * shifted_loss_mask

                    # Compute KL (K3 from http://joschu.net/blog/kl-approx.html)
                    # 1. Compute log(r) = log(pi_ref) - log(pi_theta)
                    log_r = torch.clamp(batch_ref_logprobs - policy_model_logprobs, max=2.0)

                    # 2. Compute r = pi_ref / pi_theta
                    r = torch.exp(log_r)

                    # 3. Apply the token-level k3 formula: (r - 1) - log(r)
                    kl_per_token = (r - 1.0) - log_r

                    # Calculate per token objective
                    ratio = torch.exp(policy_model_logprobs - batch_old_logprobs)
                    unclipped_obj = ratio * batch_advantages.unsqueeze(-1)
                    clipped_obj = torch.clamp(
                        ratio * batch_advantages.unsqueeze(-1),
                        1.0 - params.importance_sampling_eps,
                        1.0 + params.importance_sampling_eps,
                    )
                    policy_obj = torch.min(unclipped_obj, clipped_obj)

                    # Note: need to reapply mask because ratio (exp(0)) * advantages makes it non-zero!
                    per_token_obj = (policy_obj - params.kl_beta * kl_per_token) * shifted_loss_mask

                    # GRPO paper normalisation: normalise each completion by its own length, mean over P*G completions.
                    # Dividing each mini-batch contribution by total_completions and accumulating gives
                    # the same gradient as processing the full batch in one forward pass.
                    seq_lengths = shifted_loss_mask.sum(dim=1)
                    seq_objs = per_token_obj.sum(dim=1) / seq_lengths
                    loss = -seq_objs.sum() / total_size
                    loss.backward()

                    # Accumulate metrics
                    acc_loss += loss.item()
                    acc_kl += kl_per_token.sum().item()

                # Clip gradient norm, optimizer + LR step
                unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), params.max_grad_norm
                ).item()
                optimizer.step()
                wsd_scheduler.step()

                # Calculate metrics
                avg_loss = acc_loss / num_batches
                avg_kl = acc_kl / loss_mask[:, 1:].sum().item()
                metrics = {
                    "step": global_step,
                    "Loss/avg_total": avg_loss,
                    "Loss/avg_kl": avg_kl,
                    "unclipped_grad_norm": unclipped_grad_norm,
                    "LR": optimizer.param_groups[0]["lr"],
                }

                # Log metrics to wandb + print
                if grpo_iteration == 1:
                    wandb.log(metrics | dataset_metrics)
                else:
                    wandb.log(metrics)

                metrics |= dataset_metrics

                tree = Tree(
                    f"[bold yellow] Global step {global_step}, "
                    f"iteration {grpo_iteration}/{params.num_grpo_iterations} [/bold yellow]"
                )

                for k, v in metrics.items():
                    if k == "step":
                        continue

                    if isinstance(v, float):
                        formatted_val = f"{v:.2e}" if (0 < v < 1e-3) else f"{v:.4f}"
                    else:
                        formatted_val = str(v)
                    tree.add(f"[cyan]{k}[/cyan]: {formatted_val}")

                console.print(tree)

                # TODO: Save checkpoints


@app.function(
    gpu=params.gpu,
    timeout=params.timeout,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    volumes={"/root/.triton": kernel_volume},
)
def train_grpo_with_wandb():
    wandb_run = wandb.init(project=params.wandb_project, name=params.wandb_run_name, config=asdict(params))

    try:
        train_grpo(wandb_run)
    finally:
        wandb.finish()
