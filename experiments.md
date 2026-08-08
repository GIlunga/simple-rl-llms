Note: this is just an experiment scratchpad, full writeups will be on gilunga.github.io.

## Tasks
Different difficulties:
- 1-10, 5 turns (easy) 
- 1-20, 5 turns (medium)
- 1-50, 10 turns (hard)
- 1-100, 10 turns (very hard)

For each difficulty, run:
- Baseline (0 LR)
- GRPO no-think (32 max)
- GRPO think (512 max?) TODO: might require more prompting for less thinking?

As the difficulty increases, if needed, can try:
- Increasing number of completions per prompt
- Increasing group size
- Increasing max tokens
- Bigger models

Other possible experiments:
- Starting easy and increasing difficulty
- Binary rewards