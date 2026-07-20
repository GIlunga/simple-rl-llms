# Simple RL for LLMs

The goal of this project is to implement the GRPO algorithm and test it on simple environments at small scale using Modal.

## Setup
- Install uv, create env with uv sync, activate env
- Run `modal setup`
- Add a huggingface token to Modal secrets with the name `huggingface-secret`
- Update `app.function` or image definition as needed
- Run with `modal run train_grpo.py`

Note: the first inference and training forward will be slower due to kernel compilation but kernels are stored in a volume that persists across runs!