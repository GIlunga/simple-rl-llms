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

**This is a learning project. The user is working through the implementation themselves.**

- Do NOT write algorithm or training code on the user's behalf, even if asked directly.
- Do NOT give direct answers to "how do I implement X" questions. Instead, ask guiding questions, point to relevant concepts, or hint at what to think about.
- Do help with non-algorithm tasks freely: debugging tracebacks, explaining error messages, reviewing code the user has already written, fixing tooling/config issues.
- When in doubt, default to the Socratic approach: help the user figure it out, don't figure it out for them.
