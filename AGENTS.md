# AGENTS.md

GRPO training for a small LLM on the "guess the number" game, run on Modal GPUs.

## Layout
- `env.py` — the guess-the-number environment (parsing, observations, termination) plus the integrated reward system (`GuessTheNumberEnv.score`) and env metrics (`env.metrics`, `compute_env_metrics`).
- `train_grpo.py` — the whole trainer (env rollouts, GRPO objective, logging, Modal image/app). All hyperparameters live in the `Parameters` dataclass at the top.
- `Notes.md` — known bugs/caveats from experiments (masking, KL, normalization).
- `pyproject.toml` — uv project; deps pinned (torch from cu126 index, modal, wandb).

## Running
- `modal run train_grpo.py` (first run is slow: Triton kernel compilation; kernels are cached in the `kernel-cache` volume at `/root/.triton`).
- Local dev: `uv sync` / `uv run`; lint with ruff (line-length 120).

## Key implementation details
- Rollout generation is sequential per prompt: `num_prompts_per_step` fresh envs, `num_outputs_per_prompt` copies of one env (copies share the same target number — `base_env.clone()`).
- `env.py` owns the game: `step` parses the last `\boxed{<int>}` per turn, returns a `StepResult` (observation/termination), and treats out-of-range and repeated guesses as invalid actions (no termination, but `invalid_action_reward` when they decide a truncated budget).
- Rewards are computed by `env.score` (not read from observations). `reward_type` selects `binary` (solved within the turn budget) or `dense` (distance partial credit); format errors reward `format_error_reward` (0.0) and terminate.
- Rewards are scored for multiple turn budgets (`reward_breakdown_turns`), all from the same trajectory, and logged as `Rewards@{budget}/*`. Training uses `params.max_turns`.
- Rollouts run to the largest budget (`max(reward_breakdown_turns)`) so smaller budgets can be derived by truncating the recorded guesses.
- Loss mask: model tokens only — system prompt, observations, and special tokens are masked out. Attention mask uses `tokens != pad_token`, with `pad_token` set to `eos` token ("`</s>`"). See `Notes.md` for the EOS-attention bug this causes.
- GRPO step: PPO-clipped importance sampling ratio + K3 unbiased KL estimator (log-ratio clamped at 2.0), advantages = group-normalized rewards.
- Gradient accumulation: batch is split into `per_device_batch_size` microbatches; each microbatch loss is divided by **total** completions (P*G), so one optimizer step == one full-batch GRPO step. Re-apply loss mask after computing the ratio (exp(0) makes unmasked positions non-zero).
- Reference model is on CPU, moved to CUDA only for logprob computation, synced from policy at each iteration.
- WSD (warmup/stable/cosine-decay) LR schedule via `LambdaLR`; grad norm clipped to 2.0.
- Metrics logged to wandb  and printed as rich trees/panels; rollouts are printed with model-generated tokens highlighted.

## Dev guidance
- Add type hints to function inputs/outputs but not for internal variables
- 