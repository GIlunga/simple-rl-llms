# Simple RL for LLMs

The goal of this project is to implement the GRPO algorithm and test it on simple environments at small scale using Modal.

## Setup
- Run `uv sync`
- Enable the git hooks: `git config core.hooksPath .githooks` (the `pre-commit` hook runs `ruff` — auto-fixing and aborting if it changes files — then `pytest`)
- Run `modal setup`
- Add a huggingface token to Modal secrets with the name `huggingface-secret`
- Add a wanbd API key to Modal secrets with the name wand-secret
- Update `app.function` or image definition as needed
- Run with `modal run src/train.py` (or locally with `uv run src/train.py`)

Note: the first inference and training forward will be slower due to kernel compilation but kernels are stored in a volume that persists across runs!