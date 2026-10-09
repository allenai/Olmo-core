# OLMo Core ⇄ MaxText checkpoints

These scripts convert model weights between OLMo Core checkpoints and
[MaxText](https://github.com/AI-Hypercomputer/maxtext) checkpoints, in both directions. They also check that the two
frameworks compute the same function from the converted weights.

- **Weights only, for now.** Optimizer state isn't converted, so the target framework starts with a fresh optimizer.
- **Every parameter is a pure re-layout** (transpose, reshape, split, concatenate, stack). Converting there and back is
  bit-exact, and both conversion scripts check that, one layer at a time, unless you pass `--skip-validation`.

The library code is in `olmo_core.nn.maxtext` and follows the Hugging Face converter in `olmo_core.nn.hf`:

| Module | What it does |
|---|---|
| `config.py` | `get_maxtext_config()` picks the MaxText decoder for an OLMo Core model config, checks that the decoder can build it, and gives the MaxText overrides that size it. |
| `convert.py` | The weight mappings, as `StateMappingTemplate`s for `olmo_core.nn.conversion.StateConverter`, grouped by OLMo Core component: embeddings and LM head, norms, attention, Kimi Delta Attention, feed-forward, shared and routed experts. |
| `checkpoint.py` | Reads and writes OLMo Core DCP checkpoints (`model.*` or DDP `module.*.main` keys) and MaxText Orbax checkpoints. Converts between MaxText's scanned and unscanned layouts. |
| `convert_checkpoint.py` | `convert_checkpoint_to_maxtext()` and `convert_checkpoint_from_maxtext()`. |
| `parity.py` | Compact logit summaries, comparison metrics, and `maxtext_logit_summary()`, which runs a converted checkpoint in MaxText. |

## Supported models

The decoder is chosen from the OLMo Core block config:

| MaxText `decoder_block` | OLMo Core models |
|---|---|
| `olmo3` | Dense `reordered_norm` blocks (norms on the attention and feed-forward outputs): attention with RoPE and global QK norm, a SwiGLU feed-forward, and either no sliding window or a `[w, w, w, -1]` pattern. |
| `olmoe3` | Peri-norm blocks with a normalized embedding. Kimi Delta Attention, with a gated, NoPE, scalable-softmax full-attention layer at the end of each cycle. A dense first layer, then latent routed experts plus a shared expert. |

Anything else raises `NotImplementedError`. To support another architecture:
1. Add its decoder to `MaxTextDecoderBlock`.
2. Add its scan layout and base config in `config.py`.
3. Add its mapping builders in `convert.py`.

## Converting

The scripts need torch, OLMo Core, and a MaxText checkout that has the model's decoder, all in one environment. CPU JAX
is enough. Checkpoints are held in memory in fp32 (about 50 GB for a 12B-parameter model), so convert large models on a
high-memory machine.

```bash
# OLMo Core -> MaxText (scan_layers=True layout; add --unscanned for scan_layers=False).
# Logs the MaxText model_name and overrides to load it with.
python src/examples/maxtext/convert_checkpoint_to_maxtext.py -i <olmo-core-step-dir> -o <maxtext-out>
# MaxText: load_parameters_path=<maxtext-out>/0/items

# MaxText -> OLMo Core. Uses the olmo_core_config.json saved next to a converted checkpoint,
# or pass --olmo-core-config.
python src/examples/maxtext/convert_checkpoint_from_maxtext.py -i <maxtext-run>/checkpoints/<step>/items -o <olmo-core-out>
```

Converted OLMo Core checkpoints contain `model_and_optim/` (DCP, with `model.<name>` keys at full shape) and
`config.json`. This is the same layout the Hugging Face importer writes.

### Setting up the environment

1. Install MaxText's `src/dependencies/requirements/generated_requirements/decoupled-requirements.txt` without the
   `cuda*`/`nvidia*`/`triton` lines; it installs on CPU that way.
2. Add `jax==jaxlib==0.11.2`, since that file leaves JAX out.
3. `pip install --no-deps -e <maxtext>`.
4. Install CPU torch and `-e <OLMo-core>`.

`pytest src/test/nn/maxtext` then also runs `maxtext_test.py`, which checks against MaxText itself:
- the converted trees against MaxText's own parameter shapes, for both decoders
- a small dense model's logits in both frameworks, on CPU.

## Checking parity

`reference_logits.py` runs the model in OLMo Core in strict fp32 and saves a logit summary. `maxtext_logits.py` runs
the converted checkpoint in MaxText on the same tokens. It checks that every loaded parameter equals the checkpoint's,
then compares the logits.

```bash
# 1. A random model from a small config; save it as a checkpoint, plus reference logits.
#    Use --checkpoint <olmo-core-step-dir> instead of --config for a real checkpoint.
python src/examples/maxtext/reference_logits.py \
    --config src/test/nn/maxtext/fixtures/hybrid_moe_small_config.json \
    --save-checkpoint /tmp/small/olmo --random-tokens 2 192 --exact-kda --output /tmp/small/ref.npz

# 2. Convert, then compare in MaxText (CPU is fine).
python src/examples/maxtext/convert_checkpoint_to_maxtext.py -i /tmp/small/olmo -o /tmp/small/mt
JAX_PLATFORMS=cpu python src/examples/maxtext/maxtext_logits.py \
    --checkpoint /tmp/small/mt/0/items --reference /tmp/small/ref.npz
```

Dense models run on CPU with `--device cpu`.

Models with Kimi Delta Attention need a GPU and the official OLMo Core image, because generic images are missing
pieces:
- OLMo Core's KDA needs flash-linear-attention's Triton kernels.
- OLMo Core's MoE path without expert parallelism needs Transformer Engine (`moe_permute`).
- The torch wheel on PyPI may be built for a newer CUDA than the node's driver supports.

To run MaxText on GPU:
- **Driver:** the node's driver has to work with JAX's CUDA build. A B300 (driver 590) works with
  `jax[cuda13]==0.11.2` and `nvidia-cublas>=13.2`. A100 nodes with driver 570 fail at cuDNN init.
- **Precision:** set `JAX_DEFAULT_MATMUL_PRECISION=highest`, because raw `einsum`s otherwise use TF32.
- **Kernels:** pass `-- megablox=False attention=dot_product` to skip the Pallas GPU kernels that this JAX version
  breaks.

### Gotchas

- **TF32 in the reference.** On Ampere and newer, Triton's fp32 `tl.dot` defaults to TF32, and FLA's KDA kernels
  hard-code TF32 for their triangular solve. That alone moves logits by about 1e-2 relative. `reference_logits.py`
  sets `TRITON_F32_DEFAULT=ieee`.
- **FLA's fused KDA kernel is about 1e-3 from float64 in isolation**, even with IEEE dots. `--exact-kda` swaps in a
  float64 naive recurrence with the same gate and L2 norm, for a clean reference. It only handles single-document rows.
- **MaxText samples EMO expert pools unless `model_call_mode=inference`.** `enable_dropout=False` does not turn this
  off. OLMo Core only samples in `train()` mode. `maxtext_logit_summary()` pins the pool size to the eval pool. The
  same applies to eval losses that MaxText reports during training.
- **MaxText's chunked KDA path (`gdn_chunk_size` > 0) isn't exact.** The scan path (`gdn_chunk_size=0`) is. Use the
  scan path for parity checks, or a sequence length that isn't a multiple of the chunk size.
- **KDA's q/k L2 norm differs between the frameworks.** MaxText's pure-JAX path computes
  `x * rsqrt(max(sum(x²), 1e-12))`. FLA computes `x * rsqrt(sum(x²) + 1e-6)`. This only shows up when q and k are
  small, e.g. in a randomly initialized model.
- **OLMoDDP's `init_weights()` casts the model to its bf16 training dtype.** `reference_logits.py` casts it back to
  fp32 after random init.
- **The official OLMo Core image sets `UV_SYSTEM_PYTHON=1`,** so `uv pip install` goes into `/opt/conda` even with
  `VIRTUAL_ENV` set. Pass `--python <venv>/bin/python` to install into the venv.

## Not done yet

- Optimizer state (Adam moments and step), in both directions, for a true resume.
- Tied embeddings, scaled RoPE (YaRN) for the `olmo3` decoder, and pre-norm blocks.
- Parity checks with packed documents (several per row) and with production settings (bf16, megablox) on TPU.
