# vLLM follow-ups

State as of the `adding_vllm` branch: collocated in-process vLLM, checkpoint-layout weight sync
(all 320 HF params, fused/stacked projections), text-only engine (`language_model_only=True`),
native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`, init ~20 s), cross-target eval in one batch,
and a working 2-step training smoke. Eval-only: `num_iterations=0`, `run_test_set=True`.

## P1

### 1. Weight-sync throughput at larger models
`VLLMRolloutEngine.sync_weights` sends 320 `(checkpoint_name, tensor)` pairs through
`llm.collective_rpc("reload_weights", args=(weights,))`. At 0.8B this is ~0.15 s/step; at 7B the
pickled payload is ~14 GB and may dominate step time and host memory.

- Benchmark sync time and peak host/GPU memory after each step at the next model size.
- If it costs >~1 s/step or spikes memory, call the in-process model runner directly with a
  generator instead of pickling the tensor list. Do not use vLLM's IPC transfer backend: it fails
  in-process (`invalid device context`).
- Acceptance: measured sync time at the new size; no host-memory blowup; perturbation test still
  shows `q_proj`, `gate_proj`, and `in_proj_qkv` changes reach vLLM.

### 2. Eval scale guard
`eval_rollout_batch_size=None` submits every eval env in one call. Today that is 160 envs,
which fits vLLM's default `max_num_seqs=256` and the 146,227-token KV cache. Worst-case rollout
length at current settings is ~568 tokens (`123 prompt + 5×64 + 5×25`).

- At `eval_num_rollouts=32` and 10 targets (320 envs), requests queue past `max_num_seqs`.
- Either set a KV-safe `eval_rollout_batch_size` (~230) or raise `max_num_seqs`/KV only for
  eval-only runs. Add a guard/log so this does not silently degrade.
- Acceptance: 320+ envs run without preemption warnings or OOM; document the chosen cap.

### 3. Sampling parity with the HF fallback
The HF path uses `top_k=50` by default; vLLM uses `top_k=-1`, and seeds are not pinned per request.
Cross-backend A/B comparisons can therefore differ for reasons unrelated to weights.

- Decide whether to pin `top_k`/`top_p`/`seed` explicitly in both paths (e.g. `top_k=50` in
  `SamplingParams` or `top_k=0` in HF) or document that exact draws differ.
- Acceptance: one agreed sampling config; reproducible eval when `seed` is pinned.

## P2

### 4. `enforce_eager=False` experiment
Torch compile + CUDA graphs may give ~5–20% decode throughput. Costs: compile cache is
persistable, CUDA graph capture is per process (adds startup), extra graph memory, and weight
reload must stay graph-compatible.

- Run the same eval with eager on/off; compare generation time, init time, and memory.
- Keep eager if the graph-capture cost dominates short runs; consider graphs only for long training.

### 5. FlashInfer sampler re-enable (only if needed)
The FlashInfer sampler was disabled because its JIT took ~90 s during startup `profile_run`.
Native sampling measured equal at 160×248k vocab.

- If larger batch/vocab makes sampling material, mount `/root/.cache/flashinfer` as a Modal volume
  and set `VLLM_USE_FLASHINFER_SAMPLER=1`; the JIT then happens once per cache population.
- Measure sampler time at the target scale before switching back.

### 6. Unit tests without a GPU
- `generate_for_envs`: fake `_generate_chunk` and assert `max_concurrency` semantics
  (`None`/0 = all, int = chunks) and output ordering.
- `sync_weights`: fake LLM/`collective_rpc` and assert all 320 params are forwarded with
  checkpoint names (no matcher, no drops).

## P3

- Add the init findings (FlashInfer sampler JIT, TileLang backward compile, 20 s warm init) to
  `docs/Notes.md`.
- The remote `vllm-cache` Modal volume is now unused; delete it if desired.
- Only if a concrete need arises: pre-warm caches at image build. Eval-only init is already ~20 s.
