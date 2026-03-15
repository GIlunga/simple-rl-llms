import logging
import sys
import traceback
from copy import deepcopy

import torch
import torch.nn.functional as F
import typer
from gem.envs.game_env.guess_the_number import GuessTheNumberEnv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

logging.getLogger("httpx").setLevel(logging.WARNING)


def generate_single_rollout(env, model, tokenizer, max_rollout_tokens, n) -> tuple[torch.Tensor, int, float]:
    message_list = [
        {
            "content": "You are playing Guess The Number with the user. You have to guess the number between 1 and 10 (inclusive) within 5 turns. As you enter your guess, the user will provide you with hints such as the target number is 'higher' or 'lower'. When answering, only the number that is wrapped inside \\boxed{} will be considered as your guess, for example, \\boxed{10}. Follow that exact format for your final answer.",
            "role": "system",
        },
        {"content": "Enter your first guess to start the game!", "role": "user"},
    ]

    model.eval()

    terminated = False
    truncated = False

    inputs = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    first_input_len = inputs["input_ids"].shape[1]

    # Iterate multi-step env
    while not terminated and not truncated:
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_rollout_tokens,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Env step
        text_response = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        observation, reward, terminated, truncated, _ = env.step(text_response)

        # Add new text
        message_list.extend([{"role": "assistant", "content": text_response}, {"role": "user", "content": observation}])

        # TODO: Can we avoid re-tokenizing the previous text
        inputs = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

    if n < 4:
        print("---")
        print(message_list)
        print(env.game_number)
        print()

    return output_tokens.detach().cpu(), first_input_len, reward


def get_rollouts(
    base_env, policy_model, tokenizer, max_rollout_tokens, num_prompts_per_step, num_completions_per_prompt
):
    """Simple sequential multiple rollout generation for multiple prompts. No batching"""
    token_seq_lst = []
    initial_input_len_lst = []
    reward_lst = []
    advantage_lst = []

    for _ in range(num_prompts_per_step):
        base_env.reset()
        # Required to maintain same target for guess the number. May not be needed in other envs
        # env copies share same target number
        env_copies = [deepcopy(base_env) for _ in range(num_completions_per_prompt)]
        group_rewards = []
        for i, env in enumerate(env_copies):
            token_seq, initial_input_len, reward = generate_single_rollout(
                env, policy_model, tokenizer, max_rollout_tokens, i
            )

            token_seq_lst.append(token_seq.squeeze())
            initial_input_len_lst.append(initial_input_len)
            group_rewards.append(reward)

        group_rewards = torch.tensor(group_rewards)
        advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)

        reward_lst.append(group_rewards)
        advantage_lst.append(advantages)

    # shape: B x T
    # TODO: Incorrect loss mask - multi turn env so the observations need to be masked!
    token_seqs = torch.nn.utils.rnn.pad_sequence(token_seq_lst, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = token_seqs != tokenizer.pad_token_id
    positions = torch.arange(token_seqs.shape[1], device="cpu")
    loss_mask = positions >= torch.tensor(initial_input_len_lst).unsqueeze(-1)
    loss_mask &= attn_mask

    return token_seqs, loss_mask, attn_mask, torch.cat(reward_lst), torch.cat(advantage_lst)


def init_wandb(**args):
    return wandb.init(
        entity="gilunga-personal",
        project="Simple RL for LLMs",
        config=args,
    )


def train_grpo(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    num_steps: int = 2,
    num_prompts_per_step: int = 1,
    num_completions_per_prompt: int = 4,
    num_iterations_per_step: int = 1,  # Train multiple times on same data (iterative GRPO in R1)
    ref_model_sync_every_n_steps: int = 4,
    learning_rate: float = 1e-5,  # TODO: check what is usually used
    per_device_batch_size: int = 2,  # TODO: only supporting single device for now
    max_tokens_per_turn: int = 256,
    gradient_accumulation_batches: int = 1,
    log_to_wandb: bool = False,
):
    # TODO: log some completions
    # TODO: max_rollout_tokens is actually max_tokens per turn
    # TODO: need to double check if generation only tokens or not
    # TODO: add gradient clipping
    wandb_run = init_wandb(**locals()) if log_to_wandb else None
    # It adds a rich hook that prints locals() on each exception which is annoying
    sys.excepthook = lambda *args: traceback.print_exception(*args)

    policy_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_env = GuessTheNumberEnv(min_number=1, max_number=10, max_turns=5)

    # TODO: warmup
    optimizer = torch.optim.AdamW(policy_model.parameters(), learning_rate)

    # TODO: ignoring KL and reference model for now
    for step in tqdm(range(num_steps), desc="Training", unit="step"):
        # if step % ref_model_sync_every_n_steps == 0:
        #     reference_model = policy_model  # .copy()??

        # Generate dataset for this step (num_prompts * num_completions_per_prompt)
        token_seqs, loss_mask, attn_mask, rewards, advantages = get_rollouts(
            base_env, policy_model, tokenizer, max_tokens_per_turn, num_prompts_per_step, num_completions_per_prompt
        )

        policy_model.train()

        # Slice according to batch size per device (single device for now!)
        total_size, max_seq_len = token_seqs.shape
        num_batches = (total_size + per_device_batch_size - 1) // per_device_batch_size

        # Train multiple times on the same batch
        # TODO: I think gradient accumulation logic is wrong
        for iteration_in_step in range(num_iterations_per_step):
            acc_loss = 0.0

            # TODO: missing logprobs for importance sampling!

            for batch_idx in range(num_batches):
                if batch_idx % gradient_accumulation_batches == 0:
                    optimizer.zero_grad()

                # Get batch data
                start_idx = batch_idx * per_device_batch_size
                end_idx = (batch_idx + 1) * per_device_batch_size

                batch_inputs = token_seqs[start_idx:end_idx].cuda()
                batch_attn_mask = attn_mask[start_idx:end_idx].cuda()
                batch_loss_mask = loss_mask[start_idx:end_idx].cuda()
                batch_advantages = advantages[start_idx:end_idx].cuda()
                targets = batch_inputs[:, 1:]

                output = policy_model.forward(batch_inputs, attention_mask=batch_attn_mask)

                shifted_logits = output.logits[:, :-1, :]
                flat_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))
                flat_targets = targets.reshape(-1)

                # Cross entropy outputs the negative log likelihood of the sequence
                actual_batch_size = batch_inputs.shape[0]
                logprobs = -F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(
                    actual_batch_size, max_seq_len - 1
                )

                # Shift mask by 1 to match logprobs
                shifted_loss_mask = batch_loss_mask[:, 1:]
                per_token_obj = logprobs * shifted_loss_mask * batch_advantages.unsqueeze(-1)
                num_tokens = shifted_loss_mask.sum()

                # Normalize by number of tokens to get per-token objective
                policy_gradient_obj = per_token_obj.sum() / num_tokens

                loss = -policy_gradient_obj / gradient_accumulation_batches
                loss.backward()

                acc_loss += loss.item()

                # Step optimizer after accumulating enough gradients or in last batch
                if (batch_idx + 1) % gradient_accumulation_batches == 0 or (batch_idx + 1) == num_batches:
                    optimizer.step()

            # Calculate metrics
            # TODO: no need to repeat rewards on each iteration
            global_step = step * num_iterations_per_step + iteration_in_step
            avg_loss = acc_loss / num_batches
            avg_reward = rewards.mean().item()
            reward_std = rewards.std().item()
            null_or_format_rewards = 100 * (rewards <= 0).sum().item() / total_size

            tqdm.write(
                f"Step {global_step}: avg_loss={avg_loss:.2f}, rewards={avg_reward:.2f}, zero_reward_proportion={null_or_format_rewards:.2f}, reward_std={reward_std:.2f}"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "avg_loss": avg_loss,
                        "rewards": avg_reward,
                        "reward_std": reward_std,
                        "zero_reward_proportion": null_or_format_rewards,
                    },
                    step=global_step,
                )

            # TODO: log gradient norm, a sample completion to the terminal once in a while, checkpoints

        # TODO: reference model KL
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(train_grpo)
