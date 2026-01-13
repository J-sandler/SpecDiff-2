# SpecDiff Dev Legacy

This folder powers the SpecDiff diffusion+verifier pipeline. The `specdiff_decode.py` entrypoint copies the deployment inference flow, while training helpers (`distill_diffu_drafter.py`, `distill_diffu_phase3.py`, etc.) generate distillation data and fine‑tune drafter variants.
(See requirements.txt file)

## Inference quick start

Run the drafter+verifier together via `specdiff_decode.py` and provide both a diffusion model and a verifier model:

```bash
python3 specdiff_decode.py \
  --diffusion_model_name "apple/DiffuCoder-7B-cpGRPO" \
  --verify_model_name Qwen/Qwen2.5-7B-Instruct \
  --prompt "Write a function in python that implements a residual neural network. Answer: \n ```python \n" \
  --gamma 16 --k 16 --max_new_tokens 512 \
  --verify_temp 0.0
```

```bash
python3 specdiff_decode.py \
  --diffusion_model_name "./SpecDiffu-Qwen2.5-14B-(no-kv)/step-80000" \
  --verify_model_name Qwen/Qwen2.5-7B-Instruct \
  --prompt "Write a function in python that implements a residual neural network. Answer: \n ```python \n" \
  --gamma 16 --k 16 --max_new_tokens 512 \
  --verify_temp 0.0
```

To test on a benchmark (e.g., `gsm8k_prompts.txt` included here), feed the file and collect aggregated stats through the decode script’s batch mode:

```bash
python3 specdiff_decode.py \
    --diffusion_model_name apple/DiffuCoder-7B-cpGRPO \
    --verify_model_name Qwen/Qwen2.5-7B-Instruct \
    --prompts_file code_prompts.txt \
    --batch_stats_output ./code_stats.json \
    --gamma 16 \
    --k 16 \
    --max_new_tokens 512 \
    --verify_temp 0.0
```

This emits `[BatchSummary]` logs per run and writes JSON throughput/accept stats to `--batch_stats_output`. Use these results to gate the checkpoint before promoting it.


Try looking at `code_stats.json' now to observe the throughput stats. 

```bash
cat code_stats.json
```

Should expect ~3x+ speed-up over base models if running correctly (70+ tokens/sec). Recommend small context size while causal attention is in development.

### Argument notes

- `--diffusion_model_name`: path/ID for the (possibly PEFT) diffusion drafter. Local adapters are detected via `adapter_config.json`.
- `--verify_model_name`: verifier LM that judges candidate generations.
- `--prompt`: natural-language instruction that feeds both diffusion+verify tokenizers (optional chat template helpers available).
- `--gamma`/`--k`: beam/tree search size; larger values increase compute and acceptance.
- `--max_new_tokens`: maximum output length for the diffusion generator.
- `--verify_temp`: verifier temperature; set to `0.0` for greedy scoring.

Best results currently require powerful accelerators (multiple A100 nodes), and inference data that is coding/math-heavy.

## Distillation data generation

Use `distill_diffu_drafter.py` to synthesize labeled gamma-vs-verify sequences. A minimal command:

```bash
python3 distill_diffu_drafter.py \
  --drafter_model_id apple/DiffuCoder-7B-cpGRPO \
  --jsonl_path data/some_eval.jsonl \
  --output_dir /tmp/distill-data \
  --gamma 16 \
  --approx_kv_cache
```

The script logs generated masks/accept info per step and can profile GPU/CPU. Approx k-v cache mode trades some quality for faster autoregressive looping when you have a cache-friendly weights layout.

## Training a drafter

Once you have distillation data, you can train a drafter variant:

```bash
python3 distill_diffu_drafter.py \
  --jsonl_path /tmp/distill-data \
  --drafter_model_id apple/DiffuCoder-7B-cpGRPO \
  --output_dir /tmp/drafter-output \
  --epochs 1 \
  --approx_kv_cache
```

- `--approx_kv_cache` keeps training aligned with the deployment cache logic; it sets Dream-style masking so the training cache matches inference.
- Eval (while training) is currently unsupported—training logs focus on gamma/accept statistics and the loss itself.

## Testing checkpoints & benchmarks

After step 2 you can immediately evaluate the freshly trained checkpoint via `specdiff_decode.py` using a held-out prompt or a benchmark file. For example, if the drafter output directory is `/tmp/drafter-output`, run the same diffusion/verifier pairing as in inference:

```bash
python3 specdiff_decode.py \
  --diffusion_model_name /tmp/drafter-output \
  --verify_model_name Qwen/Qwen2.5-7B-Instruct \
  --prompt "Evaluate residual neural network generative behavior." \
  --gamma 16 --k 16 --max_new_tokens 512 \
  --verify_temp 0.0
```


## In-progress / TODO

- Remove redundant verifier passes during decoding to save time when tree verification could reuse cached logits.
- Proper KV-cache support is still being hardened; until we finish the wrapper, keep context lengths modest and prefer inference without `--approx_kv_cache` unless you validate the cache behavior.

Keep an eye on the git history for ongoing refinements such as faster decode paths and more robust PEFT/adapter compatibility.
