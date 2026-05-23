# Next Steps

- [x] Fix loss sign: `per_token_obj` is backwards — high-advantage actions should *decrease* loss, not increase it. Negate the objective.
- [x] Add reference model + KL penalty: policy diverges without a frozen reference model and KL regularization term.
- [ ] Add gradient clipping (e.g., `max_grad_norm=1.0`) to prevent exploding gradients from large advantage values.
- [ ] Rename `max_rollout_tokens` → `max_tokens_per_turn` to reflect that each turn can generate up to this many tokens (total rollout can be much larger).
- [ ] Implement reference model sync
- [ ] Add checkpointing / model saving