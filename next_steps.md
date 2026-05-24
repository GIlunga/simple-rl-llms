# Next Steps

- [x] Fix loss sign: `per_token_obj` is backwards — high-advantage actions should *decrease* loss, not increase it. Negate the objective.
- [x] Add reference model + KL penalty: policy diverges without a frozen reference model and KL regularization term.
- [ ] Add gradient clipping (e.g., `max_grad_norm=1.0`) to prevent exploding gradients from large advantage values.
- [x] Implement reference model sync
- [ ] Add importance sampling
- [ ] Add checkpointing / model saving
- [ ] Add pass@K
- [ ] Add baselines - untrained model, binary search
- [ ] TRL implementation