# CLAUDE.md

## Project

Implementing GRPO (Group Relative Policy Optimization) from scratch to fine-tune a small LLM (`Qwen/Qwen3.5-0.8B`) on a simple environment (`GuessTheNumber`). The goal is deep understanding of RL training dynamics — not production tooling.

## Commands

```bash
uv run python scripts/train_grpo.py   # run training
uv run ruff check . && uv run ruff format .  # lint
uv run pytest                          # tests
```

## Behavioral Instructions
This is a learning project. When writing algorithm or training code, explain your reasoning and design decisions before implementing. When I ask why something works a certain way, give thorough explanations. Don't just dump code — walk me through it.
