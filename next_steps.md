# Next Steps

- [x] Fix loss sign: `per_token_obj` is backwards — high-advantage actions should *decrease* loss, not increase it. Negate the objective.
- [x] Add reference model + KL penalty: policy diverges without a frozen reference model and KL regularization term.
- [ ] Add gradient clipping (e.g., `max_grad_norm=1.0`) to prevent exploding gradients from large advantage values. Also add a counter/print for when the clipping happens
- [x] Implement reference model sync
- [ ] Improve logging to work on modal + add back wandb
- [ ] Fix max tokens per turn (rename variable)
- [ ] Log KL
- [ ] Add importance sampling
- [ ] Add pass@K
- [ ] Allow thinking
- [ ] Force boxed start with no think
- [ ] Add baselines - untrained model, binary search
- [ ] Run longer, log metrics to wandb
