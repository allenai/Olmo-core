# OLMo Core ⇄ MaxText checkpoints

These scripts convert model weights, and optionally the AdamW optimizer state, between OLMo Core checkpoints and
[MaxText](https://github.com/AI-Hypercomputer/maxtext) checkpoints, in both directions.

- **Weights by default; AdamW state with `--optimizer-state`.** Without it, the target framework starts with a fresh
  optimizer. See [Optimizer state](#optimizer-state).
- **Every parameter is a pure re-layout** (transpose, reshape, split, concatenate, stack). Converting there and back is
  bit-exact, and both conversion scripts check that, one layer at a time, unless you pass `--skip-validation`.

The library code is in `olmo_core.nn.maxtext` and follows the Hugging Face converter in `olmo_core.nn.hf`:

| Module | What it does |
|---|---|
| `config.py` | `get_maxtext_config()` picks the MaxText decoder for an OLMo Core model config, checks that the decoder can build it, and gives the MaxText overrides that size it. |
| `convert.py` | The weight mappings, as `StateMappingTemplate`s for `olmo_core.nn.conversion.StateConverter`, grouped by OLMo Core component: embeddings and LM head, norms, attention, Kimi Delta Attention, feed-forward, shared and routed experts. |
| `checkpoint.py` | Reads and writes OLMo Core DCP checkpoints (`model.*` or DDP `module.*.main` keys, and AdamW state) and MaxText Orbax checkpoints (parameters, and `opt_state`). Converts between MaxText's scanned and unscanned layouts, and builds the MaxText config a converted checkpoint loads into. |
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

### Optimizer state

Pass `--optimizer-state` to either script to also convert the AdamW moments and step count. OLMo Core's AdamW and
optax's `adamw` (MaxText's default `opt_type`) keep the same moments and bias-correct them from the same step count, so
the moments go through the same re-layouts as the weights, with nothing rescaled.

- **OLMo Core -> MaxText** reads the OLMoDDP train module's `module.<name>.{exp_avg,exp_avg_sq,step}` entries or the
  FSDP train module's flattened `optim.state.<name>.*` entries. It writes a full training state (`params`, `step`,
  `opt_state`), so load it with `load_full_state_path=<maxtext-out>/0/items` instead of `load_parameters_path`. Both
  optax counts are set to the step, so MaxText's learning rate schedule resumes there.
- **MaxText -> OLMo Core** writes the moments as the OLMoDDP train module's flat `module.<name>.*` entries next to the
  `model.*` weights. OLMoDDP resumes with them; the FSDP train module loads only the weights. Pass
  `--optim-key-prefix` if your OLMoDDP model's parameter names aren't prefixed with `module.`.

Only the optimizer state is converted. Set the hyperparameters (`adam_b1`/`adam_b2`/`adam_eps`/`adam_weight_decay`
vs. OLMo Core's `betas`/`eps`/`weight_decay`) and the learning rate schedule to match on the target side yourself.
These aren't converted:
- MaxText's `adam_pax` and Muon, MaxText's `skip_step_on_spikes` wrapper (it's read through, but not written), and its
  `trainable_parameters_mask`.
- The skip-step rolling loss and grad-norm statistics, on either side. They refill within one window.
- OLMo Core's persistent model buffers (`model_buffer.*`, e.g. the router's `score_bias`), as with weights-only
  conversion.
- The data loader position and trainer state.

The moments are held in memory in fp32 next to the weights, so expect about three times the memory of a weights-only
conversion.

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

- An end-to-end resume test with converted optimizer state: matching losses for the first steps after resuming in the
  other framework.
- Tied embeddings, scaled RoPE (YaRN) for the `olmo3` decoder, and pre-norm blocks.
- Logit-parity checks of converted checkpoints beyond the small CPU test in `src/test/nn/maxtext/maxtext_test.py`.
