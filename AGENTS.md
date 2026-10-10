# AGENTS.md

GRPO training for a small LLM on the "guess the number" game, run on Modal GPUs.

## Layout
- `src/guess_env.py` — the guess-the-number environment (parsing, observations, termination) plus the integrated reward system and per-rollout metrics (`GuessTheNumberEnv.rollout_metrics`) and batch aggregation (`aggregate_env_metrics`, `aggregate_quality_metrics`). Env settings are passed directly to `GuessTheNumberEnv(...)`.
- `src/grpo.py` — the trainer (env rollouts, GRPO objective, logging). All hyperparameters are passed in as a `Parameters` argument; no Modal/wandb-init code here.
- `src/train.py` — the entrypoint. Holds the `Parameters` dataclass, constructs the `GuessTheNumberEnv`, the Modal image/app, and both the local and Modal run paths. Calls `grpo.train(params, base_env, wandb_run)`.
- `src/build_helpers.py` — `MODEL_NAME` plus the `download_models` build helper used by the Modal image. Kept out of `train.py` so the image build step doesn't import the training code (see below).
- `tests/test_env.py` — pytest suite for the environment (pytest `pythonpath = ["src"]`, `testpaths = ["tests"]`).
- `docs/Notes.md` — known bugs/caveats from experiments (masking, KL, normalization).
- `pyproject.toml` — uv project; deps pinned (torch from cu126 index, modal, wandb).

## Running
- `modal run src/train.py` — runs on Modal GPUs (first run is slow: Triton kernel compilation; kernels are cached in the `kernel-cache` volume at `/root/.triton`).
- `uv run src/train.py` (or `python src/train.py`) — runs locally via the `__main__` block.
- Local dev: `uv sync` / `uv run`; test with `uv run pytest`; lint with ruff (line-length 120). A `pre-commit` hook in `.githooks/` runs ruff (auto-fixes then aborts if it changed files so you can review/re-stage) + pytest (enable once with `git config core.hooksPath .githooks`).

## Modal image / imports
- The image is built as `...uv_sync().run_function(download_models, ...).add_local_python_source("guess_env", "grpo")`. The local-source mounts must come **last** (Modal forbids build steps after `add_local_*` unless `copy=True`).
- `download_models` lives in `build_helpers.py` so the `run_function` build step imports only that module. If it lived in `train.py`, the build would import `train.py` and thus `grpo`/`guess_env` before they are mounted — the original `ModuleNotFoundError: No module named 'guess_env'`.
- `guess_env` and `grpo` must be listed in `add_local_python_source`: the entrypoint is run as a single file, so Modal only auto-mounts `src/train.py`, not sibling modules.
- Sources live in a flat `src/` (no package). `modal run src/train.py` and `python src/train.py` both put the script's directory on `sys.path`, so `add_local_python_source` still resolves `guess_env`/`grpo` by name and intra-project imports stay top-level (`from guess_env import ...`).

## Key implementation details
- Rollout generation is sequential per prompt: `num_prompts_per_step` fresh envs, `num_outputs_per_prompt` copies of one env (copies share the same target number — `base_env.clone()`).
- `src/guess_env.py` owns the game: `step` parses the last `\boxed{<int>}` per turn, returns a `StepResult` (observation/termination), and treats out-of-range and repeated guesses as invalid actions (no termination, but `invalid_action_reward` when they decide a truncated budget).
- Rewards are computed from the recorded trajectory by `env.rollout_metrics()` (not read from observations). `reward_type` selects `binary` (solved within the turn budget) or `dense` (distance partial credit); format errors reward `format_error_reward` (0.0) and terminate.
- Rewards are scored for multiple turn budgets (`reward_breakdown_turns`), all from the same trajectory, and logged as `Rewards@{budget}/*`. Training uses `params.max_turns`.
- Rollouts run to the largest budget (`max(reward_breakdown_turns)`) so smaller budgets can be derived by truncating the recorded guesses.
- Loss mask: model tokens only — system prompt, observations, and special tokens are masked out. Attention mask uses `tokens != pad_token`, with `pad_token` set to `eos` token ("`</s>`"). See `docs/Notes.md` for the EOS-attention bug this causes.
- GRPO step: PPO-clipped importance sampling ratio + K3 unbiased KL estimator (log-ratio clamped at 2.0), advantages = group-normalized rewards.
- Gradient accumulation: batch is split into `per_device_batch_size` microbatches; each microbatch loss is divided by **total** completions (P*G), so one optimizer step == one full-batch GRPO step. Re-apply loss mask after computing the ratio (exp(0) makes unmasked positions non-zero).
- Reference model is on CPU, moved to CUDA only for logprob computation, synced from policy at each iteration.
- WSD (warmup/stable/cosine-decay) LR schedule via `LambdaLR`; grad norm clipped to 2.0.
- Metrics logged to wandb  and printed as rich trees/panels; rollouts are printed with model-generated tokens highlighted.

## Dev guidance
- Add type hints to function inputs/outputs but not for internal variables
- Never run modal run yourself