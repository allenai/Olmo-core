# OLMo Core ⇄ MaxText checkpoints

These scripts convert model weights between OLMo Core checkpoints and
[MaxText](https://github.com/AI-Hypercomputer/maxtext) checkpoints, in both directions.

- **Weights only, for now.** Optimizer state isn't converted, so the target framework starts with a fresh optimizer.
- **Every parameter is a pure re-layout** (transpose, reshape, split, concatenate, stack). Converting there and back is
  bit-exact, and both conversion scripts check that, one layer at a time, unless you pass `--skip-validation`.

The library code is in `olmo_core.nn.maxtext` and follows the Hugging Face converter in `olmo_core.nn.hf`:

| Module | What it does |
|---|---|
| `config.py` | `get_maxtext_config()` picks the MaxText decoder for an OLMo Core model config, checks that the decoder can build it, and gives the MaxText overrides that size it. |
| `convert.py` | The weight mappings, as `StateMappingTemplate`s for `olmo_core.nn.conversion.StateConverter`, grouped by OLMo Core component: embeddings and LM head, norms, attention, Kimi Delta Attention, feed-forward, shared and routed experts. |
| `checkpoint.py` | Reads and writes OLMo Core DCP checkpoints (`model.*` or DDP `module.*.main` keys) and MaxText Orbax checkpoints. Converts between MaxText's scanned and unscanned layouts, and builds the MaxText config a converted checkpoint loads into. |
| `convert_checkpoint.py` | `convert_checkpoint_to_maxtext()` and `convert_checkpoint_from_maxtext()`. |

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

For a model with EMO routing, the overrides include `olmo_segment_eos_token_id` and `olmoe3_segments_for_emo_only=True`.
OLMo Core's EMO router splits documents on the EOS token, while MaxText's takes document boundaries from the data
pipeline's segment ids, so these only take effect with MaxText's OLMo pipeline (`dataset_type=olmo_grain`). With another
pipeline, make sure it numbers EOS-delimited documents the same way, or the router will form different expert pools.

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

## Not done yet

- Optimizer state (Adam moments and step), in both directions, for a true resume.
- Tied embeddings, scaled RoPE (YaRN) for the `olmo3` decoder, and pre-norm blocks.
- Logit-parity checks of converted checkpoints beyond the small CPU test in `src/test/nn/maxtext/maxtext_test.py`.
