# AGENTS.md

GRPO training for a small LLM on the "guess the number" game, run on Modal GPUs.

## Layout
- `train_grpo.py` — the whole trainer (env rollouts, GRPO objective, logging, Modal image/app). All hyperparameters live in the `Parameters` dataclass at the top.
- `Notes.md` — known bugs/caveats from experiments (masking, KL, normalization).
- `pyproject.toml` — uv project; deps pinned (torch from cu126 index, `gem-llm==0.1.0`, modal, wandb).

## Running
- `modal run train_grpo.py` (first run is slow: Triton kernel compilation; kernels are cached in the `kernel-cache` volume at `/root/.triton`).
- Local dev: `uv sync` / `uv run`; lint with ruff (line-length 120).

## Key implementation details
- Rollout generation is sequential per prompt: `num_prompts_per_step` fresh envs, `num_outputs_per_prompt` copies of one env (copies share the same target number — `deepcopy` is required).
- Reward is parsed from the model output's last `\boxed{<int>}` (see `GuessTheNumberEnv.step` in `gem-llm`); partial credit for final distance to target, penalties for invalid/repeated/out-of-range guesses.
- Loss mask: model tokens only — system prompt, observations, and special tokens are masked out. Attention mask uses `tokens != pad_token`, with `pad_token` set to `eos` token ("`</s>`"). See `Notes.md` for the EOS-attention bug this causes.
- GRPO step: PPO-clipped importance sampling ratio + K3 unbiased KL estimator (log-ratio clamped at 2.0), advantages = group-normalized rewards.
- Gradient accumulation: batch is split into `per_device_batch_size` microbatches; each microbatch loss is divided by **total** completions (P*G), so one optimizer step == one full-batch GRPO step. Re-apply loss mask after computing the ratio (exp(0) makes unmasked positions non-zero).
- Reference model is on CPU, moved to CUDA only for logprob computation, synced from policy at each iteration.
- WSD (warmup/stable/cosine-decay) LR schedule via `LambdaLR`; grad norm clipped to 2.0.
- Metrics logged to wandb  and printed as rich trees/panels; rollouts are printed with model-generated tokens highlighted.

## Dev guidance
- Add type hints to function inputs/outputs but not for internal variables
- 