# OLMoE3 (OLMo 3.5) checkpoints: OLMo Core ⇄ MaxText

Converts OLMo 3.5 (OLMoE3) weights between OLMo Core and MaxText's `olmoe3` decoder
(the `olmo35-rebase` branch of MaxText), and checks that the two frameworks compute the same function
from them.

- Weights only, for now. Optimizer state isn't converted, so the target framework starts with a fresh
  optimizer.
- Every parameter is converted with a pure re-layout (transpose, reshape, split, concatenate, stack),
  so a round trip is bit-exact.

Naming trap: MaxText's `olmo35-tiny` is the same model as OLMo Core's 810m ladder rung and hero
"small". All three have 12,496,341,632 parameters.

## Files

| File | What it does |
|---|---|
| `olmo_core/nn/maxtext/olmoe3.py` | The mapping, plus the model geometry parsed from an OLMo Core `config.json`, scan/unscan, and the MaxText config overrides for any geometry. numpy only. |
| `olmo_core/nn/maxtext/checkpoint.py` | Reads and writes OLMo Core DCP checkpoints (`model.*` or DDP `module.*.main` keys) and MaxText Orbax checkpoints. |
| `olmo_core/nn/maxtext/parity.py` | Compact logit summaries and the comparison metrics. |
| `convert_olmoe3.py` | CLI with `to-maxtext`, `to-olmo-core` and `compare` (bit-exact weight check). |
| `olmoe3_reference_logits.py` | Runs OLMo Core on GPU in strict fp32 and writes a logit summary. Can also save a random model as a checkpoint. |
| `olmoe3_maxtext_logits.py` | Loads a converted checkpoint in MaxText (CPU is fine), checks the loaded weights equal the checkpoint's, then compares logits against a reference summary. |
| `test/nn/maxtext/olmoe3_test.py` | CPU unit tests: round trips, shapes at full 810m scale, and the semantics of each re-layout. |

## Converting

The conversion scripts need torch, OLMo Core, and MaxText with JAX in one environment. CPU JAX is
enough. Full-size checkpoints are held in memory in fp32 (about 50 GB for the 810m model), so use a
high-memory machine.

```bash
# OLMo Core -> MaxText (scan_layers=True layout; add --unscanned for scan_layers=False)
python src/scripts/maxtext/convert_olmoe3.py to-maxtext <olmo-core-step-dir> <maxtext-out>
# MaxText: load_parameters_path=<maxtext-out>/0/items

# MaxText -> OLMo Core. Uses the olmo_core_config.json that to-maxtext leaves next to the
# checkpoint, or pass --olmo-core-config.
python src/scripts/maxtext/convert_olmoe3.py to-olmo-core <maxtext-run>/checkpoints/<step>/items <olmo-core-out>

# Bit-exact weight comparison of two OLMo Core checkpoints
python src/scripts/maxtext/convert_olmoe3.py compare <a> <b>
```

Converted OLMo Core checkpoints have `model_and_optim/` (DCP, `model.<name>` keys at full shape) and
`config.json`. This is the same layout the Hugging Face importer writes.

### Setting up the environment

- MaxText's `src/dependencies/requirements/generated_requirements/decoupled-requirements.txt`
  installs on CPU once the `cuda*`/`nvidia*`/`triton` lines are removed. Then add `jax==jaxlib==0.11.2`
  (the file leaves JAX out), `pip install --no-deps -e <maxtext>`, CPU torch, and `-e <OLMo-core>`.

## Checking parity

Reference side: a GPU with the official OLMo Core image. On A100 that's
`beaker://akshitab/olmo-core-tch2130cu129-sm80-2026-09-11`. Generic images won't work:

- OLMo Core's KDA needs FLA's Triton kernels.
- OLMo Core's MoE path without expert parallelism needs Transformer Engine (`moe_permute`).
- The torch 2.13 wheel on PyPI is built for CUDA 13; many nodes only have a CUDA 12.8 driver.

```bash
# 1. GPU: random model from a small config with the same structure; save it and reference logits
python src/scripts/maxtext/olmoe3_reference_logits.py \
    --config src/test/nn/maxtext/fixtures/olmoe3_parity_small_config.json \
    --save-checkpoint /tmp/small/olmo --random-tokens 2 192 --exact-kda --output /tmp/small/ref.npz

# 2. Convert, then check in MaxText (CPU)
python src/scripts/maxtext/convert_olmoe3.py to-maxtext /tmp/small/olmo /tmp/small/mt
JAX_PLATFORMS=cpu python src/scripts/maxtext/olmoe3_maxtext_logits.py \
    --checkpoint /tmp/small/mt/0/items --reference /tmp/small/ref.npz
```

For a real checkpoint, pass `--checkpoint <olmo-core-step-dir>` instead of `--config`.

To test scan-layout handling, the small config has 16 layers (two 8-layer cycles), and 192 tokens is
three KDA chunks.

Gotchas:

- **TF32 in the reference.** On Ampere and newer, Triton's fp32 `tl.dot` defaults to TF32, and FLA's
  KDA kernels hard-code TF32 for their triangular solve. That alone moved OLMo Core's logits by
  about 7e-3 relative. The reference script sets `TRITON_F32_DEFAULT=ieee`.
- **FLA's fused KDA kernel is about 1e-3 from float64 in isolation**, even with IEEE dots.
  `--exact-kda` swaps in a float64 naive recurrence with the same gate and L2 norm, for a clean
  reference. It only handles single-document rows. On the full small model the two differed by just
  1.8e-5.
- **MaxText samples EMO expert pools unless `model_call_mode=inference`.** `enable_dropout=False`
  does not turn it off. OLMo Core only samples in `train()` mode. `olmoe3_maxtext_logits.py` pins
  `emo_min = emo_max = eval pool`. Without that, scanned and unscanned MaxText looked completely
  different (KL 0.1). The same applies to eval losses MaxText reports during training.
- **The OLMoDDP `init_weights()` casts the model to its bf16 training dtype.** The reference script
  casts it back to fp32 after random init.
- **The official image sets `UV_SYSTEM_PYTHON=1`**, so `uv pip install` goes into `/opt/conda` even
  with `VIRTUAL_ENV` set. Pass `--python <venv>/bin/python` to keep things separate.

## Results so far (2026-10-08)

**CPU:**

- Round trips are bit-exact.
- Shapes match the real 810m OLMo Core model (built on `meta`) and MaxText's `jax.eval_shape` trees,
  scanned and unscanned.
- Converted checkpoints load through MaxText's `load_parameters_path`, and every loaded parameter
  equals the checkpoint's.
- Scanned and unscanned MaxText agree to about 2e-6 relative.

**GPU, small random model:** real OLMo Core on an A100 vs MaxText on CPU, both fp32, 2 × 192 tokens.

| Comparison | Max rel. logit err | Mean KL | Top-1 agreement | Mean \|Δ loss\| |
|---|---|---|---|---|
| MaxText vs OLMo Core | 8.0e-4 | 1.0e-8 | 100% | 1.1e-4 |
| Same, with MaxText's L2 norm in the reference | 7.0e-5 | 5.3e-11 | 100% | 7.5e-6 |

Per layer, measured by feeding each MaxText layer the OLMo Core input to that layer:

- **About 1e-6:** the embedding, the LM head, and the full-attention blocks (which include the routed
  MoE and shared expert).
- **About 1e-4:** the KDA blocks.

## Known differences between the frameworks

- **KDA q/k L2 norm.** MaxText's pure-JAX KDA path (`use_tokamax_kda=false`, which the v4 job uses)
  computes `x * rsqrt(max(sum(x²), 1e-12))`. OLMo Core/FLA, and MaxText's tokamax path, compute
  `x * rsqrt(sum(x²) + 1e-6)`. This accounts for the 8e-4 above. It matters most when q/k are small
  (early in training) and should be fixed in MaxText's `_l2_normalize` in `olmoe3.py`.
- **MaxText's existing `tests/unit/olmo35_vs_reference_test.py` uses a different reference.** It
  checks against a standalone "partner model", not OLMo Core's modules, and their layouts differ: the
  partner model's routed down projection is `[E, latent, hidden]`, OLMo Core's `w_down` is
  `[E, hidden, latent]`.

## Not done yet

- A real checkpoint (`gs://olmo-3p5-checkpoints/external/olmo-3p5-tiny/emo/checkpoints/`):
  structure, logits on real text at long sequence lengths, and a bit-exact round trip.
- A short MaxText training run on TPU v4 from a converted checkpoint.
- The step-0 training comparison: same init and batch, EMO off. No optimizer conversion is needed,
  since AdamW starts from zero moments.
- Phase 2: optimizer state (Adam moments and step) in both directions, for a true resume.
