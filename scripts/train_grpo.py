import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
import typer
from gem.envs.game_env.guess_the_number import GuessTheNumberEnv
from rich.console import Console
from rich.text import Text
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("httpx").setLevel(logging.WARNING)

def print_masked_sequence(sequences: torch.Tensor, mask: torch.Tensor, tokenizer) -> None:
    text = Text()
    for tok, m in zip(sequences.tolist(), mask.tolist(), strict=True):
        decoded = tokenizer.decode([tok]).replace("\n", "↵")
        if m:
            text.append(decoded, style="bold green")
        else:
            text.append(decoded, style="dim")
    Console().print(text)


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

    # Thinking = False adds a thinking section and closes it immediately
    inputs_text = tokenizer.apply_chat_template(
        message_list, tokenize=False, enable_thinking=False, add_generation_prompt=True
    )

    inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
    prev_len = inputs["input_ids"].shape[1]
    output_mask = [False] * prev_len  # Mask out system prompt + special tokens
    ENDOFTEXT_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    im_end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Iterate multi-step env
    while True:
        with torch.inference_mode():
            output_dict = model.generate(
                **inputs,
                max_new_tokens=max_rollout_tokens,
                temperature=1.0,
                do_sample=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, im_end_token]
            )

        # Strip end of text
        if output_dict.sequences[0][-1] == ENDOFTEXT_TOKEN_ID:
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

    sequences = output_dict.sequences[0]
    mask = torch.tensor(output_mask)
    print()
    print(f"{terminated=}")
    print(f"{truncated=}")
    print_masked_sequence(sequences, mask, tokenizer)
    return output_dict.sequences.detach().cpu(), mask, reward


def get_rollouts(
    base_env, policy_model, tokenizer, max_rollout_tokens, num_prompts_per_step, num_completions_per_prompt
):
    """Simple sequential multiple rollout generation for multiple prompts. No batching"""
    token_seq_lst = []
    output_mask_lst = []
    reward_lst = []
    advantage_lst = []

    for _ in range(num_prompts_per_step):
        base_env.reset()
        # Required to maintain same target for guess the number. May not be needed in other envs
        # env copies share same target number
        env_copies = [deepcopy(base_env) for _ in range(num_completions_per_prompt)]
        group_rewards = []
        for i, env in enumerate(env_copies):
            token_seq, output_mask, reward = generate_single_rollout(
                env, policy_model, tokenizer, max_rollout_tokens, i
            )

            token_seq_lst.append(token_seq.squeeze())
            output_mask_lst.append(output_mask)
            group_rewards.append(reward)

        group_rewards = torch.tensor(group_rewards)
        advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)

        reward_lst.append(group_rewards)
        advantage_lst.append(advantages)

    # shape: B x T
    token_seqs = torch.nn.utils.rnn.pad_sequence(token_seq_lst, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = token_seqs != tokenizer.pad_token_id

    loss_mask = torch.nn.utils.rnn.pad_sequence(output_mask_lst, batch_first=True, padding_value=False)
    loss_mask &= attn_mask

    return token_seqs, loss_mask, attn_mask, torch.cat(reward_lst), torch.cat(advantage_lst)


def train_grpo(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    num_steps: int = 4,
    num_prompts_per_step: int = 1,
    num_completions_per_prompt: int = 8,
    num_iterations_per_step: int = 1,  # Train multiple times on same data (iterative GRPO in R1)
    ref_model_sync_every_n_steps: int = 4,
    learning_rate: float = 1e-5,  # TODO: check what is usually used
    per_device_batch_size: int = 2,  # TODO: only supporting single device for now
    max_tokens_per_turn: int = 256,
):
    # TODO: max_rollout_tokens is actually max_tokens per turn
    # TODO: need to double check if generation only tokens or not
    # TODO: add gradient clipping

    policy_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    base_env = GuessTheNumberEnv(min_number=1, max_number=10, max_turns=5)

    # TODO: warmup?
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
        for iteration_in_step in range(num_iterations_per_step):
            acc_loss = 0.0

            # TODO: missing logprobs for importance sampling!

            optimizer.zero_grad()
            for batch_idx in range(num_batches):
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

                # GRPO paper normalisation: normalise each completion by its own length, mean over P*G completions.
                # Dividing each mini-batch contribution by total_completions and accumulating gives
                # the same gradient as processing the full batch in one forward pass.
                seq_lengths = shifted_loss_mask.sum(dim=1)
                seq_objs = per_token_obj.sum(dim=1) / seq_lengths
                loss = -seq_objs.sum() / total_size
                loss.backward()

                acc_loss += loss.item()

            optimizer.step()

            # Calculate metrics
            # TODO: no need to repeat rewards on each iteration
            global_step = step * num_iterations_per_step + iteration_in_step
            avg_loss = acc_loss / num_batches
            avg_reward = rewards.mean().item()
            reward_std = rewards.std().item()
            null_or_format_rewards = 100 * (rewards <= 0).sum().item() / total_size

            tqdm.write(
                f"Step {global_step}, iteration {iteration_in_step}/{num_iterations_per_step}: avg_loss={avg_loss:.2f}, rewards={avg_reward:.2f}, zero_reward_proportion={null_or_format_rewards:.2f}, reward_std={reward_std:.2f}"
            )


            # TODO: log gradient norm, a sample completion to the terminal once in a while, checkpoints

        # TODO: reference model KL

if __name__ == "__main__":
    typer.run(train_grpo)
