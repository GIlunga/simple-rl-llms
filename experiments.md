Note: this is just an experiment scratchpad, full writeups will be on gilunga.github.io.

On task definition:
- The task is meant to have the turn count in the system prompt
- To make things more efficient, we can instead say "find the number as fast as possible" and cut it off when needed
- Then no need to train different versions nor evaluate multiple times?

On evaluating different strategies against each other:
- Randomly sampling is not the best idea, can define a specific test set
- Test sets:
  - Easy: 1-10, 6 turns (if you start in the middle, you can walk and get 100%, worst case binary search is 4)
  - Medium: 1-10, 5 turns (needs two 50/50 slices before walking)
  - Hard: 1-10, 4 turns (binary search or luck)
  - Hard with different numbers: 20-30, 4 turns
- As mentioned in task definition, we can evaluate all test sets (except the last one) in one go, by allowing the model to go up to 6 turns and count wins with different max turns

- Number of rollouts should be large, e.g., 32, and then we can estimate pass@K for K<32
- Results should include a line plot of pass@K, for a few different Ks, for each number
- Also accuracy per number
- Run at the end of training, too expensive to run while training, unless inference is optimized.

On what to experiment with:
- No think vs think
- Varying number of groups and number of rollouts in training
- Binary vs dense rewards vs task specific rewards (e.g., rewards for cutting search space)

## Roadmap
- [ ] Without training, evaluate no think model on all test sets
- [ ] Train no think model, evaluate it on all test sets