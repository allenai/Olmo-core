# muP verification report

This document records the tests run to verify the muP implementation in OLMo-core (nanochat init
plus per-parameter Adam learning rates), what each test showed, and what it does and does not
establish. See `docs/MUP_HANDOFF.md` for how muP is implemented and
`docs/NANOCHAT_INIT_HANDOFF.md` for the init.

## Summary

- **The implementation does what was specified.** In 10 short training runs on 8 B300 GPUs,
  covering 6 model sizes from 0.3B to 42.8B parameters (including the exact small and medium hero
  models), every check passed:
  - every parameter's learning rate equalled `lr * base_width / fan_in` (1x for embeddings and
    vectors), using that parameter's own fan_in;
  - that LR held at every step of the schedule (worst relative error 2.5e-8, which is fp32
    rounding);
  - the measured fp32 weight updates show that this LR is the one the optimizer actually applied.
- **The per-parameter fan_in design is visible in the results.** In the medium hero, the number
  of heads grows faster than d_model. Exactly the three output-projection roles that read the head
  space (24 tensors) got a lower LR than a single model-wide width multiplier would give. Every
  other projection matched the d_model-based value.
- **Coordinate check.** On the hero architecture, the logits' movement grows with width under
  the standard parameterization (exponent +0.51) and stays flat under muP (+0.01). Hidden layers
  are flat under both, most likely because the hero's peri-norm blocks normalize them, so on this
  architecture the check is only informative at the logits.
- **Not established:** that a tuned LR actually transfers across sizes (needs real data and an
  LR sweep), the production `rowwise_nvshmem` expert-parallel path, and checkpoint resume. The
  muP design scales width only; depth, expert count and weight decay are not scaled.

## The design being verified

muP has two parts:

- **Init:** nanochat init, std = `(1 / sqrt(fan_in)) * min(1, sqrt(fan_out / fan_in))` for hidden
  matrices (including `lm_head`), std 1.0 for embeddings, `1 / sqrt(kernel_size)` for depthwise
  convs, and no zero-init. See `InitMethod.nanochat` in `src/olmo_core/nn/transformer/init.py`.
- **Learning rates:** the Adam rule `lr_l = lr * base_width / fan_in_l` for every hidden matrix,
  with multiplier 1 for embeddings and vector-like parameters (norm gains, biases, depthwise
  convs, `A_log`, `dt_bias`, `ssmax_scale`). Configured with
  `MuPConfig(base_width=...)` on the optimizer config.
  - The multipliers come from `mup_lr_multipliers` in `src/olmo_core/optim/mup.py`.
  - The fan_in values come from the shape registry `named_param_shapes` in
    `src/olmo_core/nn/transformer/mup.py`.

### Design decision: each parameter is scaled by its own fan_in

Every matrix's multiplier is `base_width` divided by **that matrix's own fan_in**, with
`base_width` a fixed constant (1024 in these tests, the small hero's d_model). This combines two
choices:

- **Per-parameter fan_in, not one model-wide multiplier.** Implementations that follow the
  Cerebras/EleutherAI practitioner's guide compute one `width_multiplier = d_model / base_width`
  and divide every hidden matrix's LR by it. We use each matrix's own fan_in instead.
- **A fixed `base_width` constant, not a base model.** The alternative we considered was anchoring
  each parameter to its fan_in in a base model config, as Microsoft's `mup` library does with
  `set_base_shapes`. That puts every multiplier at exactly 1 in the base model, but it needs a
  base model with the same depth and block layout. With a fixed constant, multipliers at the base
  width are 1 only for matrices whose fan_in equals `base_width` (see below).

Per-parameter fan_in matters because the hero's dimensions don't all scale together:

- Going from small to medium, d_model grows 1.5x (1024 to 1536) but the number of heads doubles
  (8 to 16). Head dim is fixed at 128, so the attention head space grows from 1024 to 2048.
- Matrices that read the head space (the attention and KDA output projections) therefore have a
  fan_in that grows faster than d_model, and they should get a correspondingly smaller LR.
- Some fan_ins don't scale at all (the KDA low-rank projections read a fixed 256-wide
  bottleneck), so their LR should stay constant across sizes.

Expected consequences of this choice, which the results below confirm:

- Projections from the residual stream (fan_in = d_model) get `base_width / d_model`.
- The full-attention `w_out` (fan_in = heads x 128) and KDA `w_out` (fan_in = heads x 128 x 2)
  get multipliers set by the head count, not by d_model.
- Fixed-fan_in matrices (`f_proj_2`, `g_proj_2`, fan_in 256) get `1024 / 256 = 4` at every size.
- At d_model = base_width, the multipliers are not all 1. This comes from anchoring with a fixed
  constant rather than a base model. At d1024:
  - KDA `f_proj_2` and `g_proj_2` get 4x;
  - `latent_up_proj` and routed `w_up_gate` (fan_in d/2) get 2x;
  - KDA `w_out` (fan_in 2d) gets 1/2;
  - the dense block's `w_down` (fan_in 8d) gets 1/8.

  So muP at the base width is not the standard parameterization, and the base LR must be tuned
  under muP rather than reused from a standard run. The init change requires that anyway.

## What was tested

### Test harness

Script: `src/examples/mup/hero_lr_check.py`.

- `run <size> <mup|sp>`: trains one model under torchrun on 8 GPUs and writes a JSON result file.
- `report`: reads all result files and prints the tables below. It exits non-zero if any hard
  check failed.

Each run uses the production training stack, not a simplified loop:

- the real `Trainer`, `OLMoDDPTrainModule` and `OLMoDDPOptimizer`;
- the production optimizer settings: lr 8e-4, weight decay 0.1 (0 for embeddings), betas
  (0.9, 0.95), the production group overrides, `compile=True`, `sigma_factor=6`,
  `use_distributed=True`, gradient clipping at 1.0;
- DDP with fp32 gradient accumulation and reduction, and z-loss 1e-5.

Run settings, and where they differ from production:

- 20 steps with `WSD(warmup=5, decay=5)`, so the LR check covers warmup, stable and decay.
- Random tokens from `RandomInstanceSource` (dolma2 tokenizer, documents about 1024 tokens with
  EOS boundaries for the EMO router).
- Sequence length 4096, 16 sequences per global batch (65,536 tokens), gradient accumulation 2.
- Not production: `compile_model=False`, so forward hooks can capture activations for the
  coordinate check.
- Not production: KDA uses the FLA kernels (`use_cute_kernel=False`), because `kernel_fun` is not
  installed on this machine.
- Per-head QK-norm gains are on, as in the hero runs.
- muP runs use `InitMethod.nanochat` plus `MuPConfig(base_width=1024)`. Standard-parameterization
  (SP) runs use the production `InitMethod.normal` (std 0.02) and one LR for every group.

### Model sizes

| Size | d_model | Heads | Layers | Experts (top-k) | Parameters | Tensors | Parallelism | Runs |
|---|---|---|---|---|---|---|---|---|
| d512 | 512 | 4 | 8 | 64 (8) | 309,530,656 | 207 | DDP | muP, SP |
| d1024 | 1024 | 8 | 8 | 64 (8) | 1,022,789,952 | 207 | DDP | muP, SP |
| d1536 | 1536 | 12 | 8 | 64 (8) | 2,139,779,680 | 207 | DDP | muP, SP |
| d2048 | 2048 | 16 | 8 | 64 (8) | 3,660,499,840 | 207 | DDP | muP, SP |
| small (real hero) | 1024 | 8 | 16 | 512 (16) | 12,496,341,632 | 415 | DDP | muP |
| medium (real hero) | 1536 | 16 | 24 | 512 (16) | 42,759,806,592 | 623 | DDP + EP=8 | muP |

The four hero-like sizes reuse the hero's block builders from
`src/examples/olmo_ddp/olmoe3_small_medium_models.py`. Their layer pattern:

- block 0: KDA with the dense 8d shared MLP;
- block 7: gated full attention with QK-norm;
- other blocks: KDA with latent MoE.

They use `heads = d_model / 128`, so their head space always equals d_model. The real medium hero
is the only size where the head space (2048) differs from d_model (1536).

The medium hero does not fit under plain DDP on 8 GPUs, so it ran with expert parallelism across
all 8 GPUs (64 experts per GPU). It used the synchronized all-to-all path (`sync_1d`) because the
production `rowwise_nvshmem` path needs a CUDA extension that can't be built here (no `nvcc`).

### Checks

All checks run inside a trainer callback (`MuPLRCheck`). Checks 1 to 3 are hard assertions;
check 4 is reported only.

1. **Group membership and multipliers** (before training).
   - Every trainable parameter must be in exactly one optimizer group.
   - Each group's multiplier must equal an expected value computed independently in the script
     from the parameter's name and tensor layout (`expected_fan_in`), not from the muP code's
     shape registry:
     - `nn.Linear` weight `[out, in]`: fan_in = `in`;
     - routed-expert `w_up_gate` `[E, 2H, D]`: fan_in = `D`;
     - expert `w_down` `[E, H, D]`: fan_in = `H`;
     - shared-expert `w_up_gate` `[D, E * 2H]`: fan_in = `D`;
     - flattened router weight `[E * D]`: fan_in = d_model;
     - embeddings, norms, convs and 1-D tensors: multiplier 1.
   - Under SP, every multiplier must be 1.
2. **LR at every step** (after each optimizer step). For every group, the LR stored in the group
   must equal `multiplier * scheduled LR` (relative tolerance 1e-6), for all 20 steps. The logged
   `optim/LR (group i)` metrics must match the stored value. That comparison covers steps 1 to
   19, because the final step's metrics are logged after the results are written.
3. **The LR reaches the weights.** The fp32 master weights are snapshotted before each optimizer
   step and diffed after it.
   - The optimizer applies `W <- W * (1 - lr * wd) - lr * u`, where `u` is Adam's normalized
     update. So `u = -(dW + lr * wd * W_old) / lr` can be recovered from the weight change, up to
     fp32 rounding, using the group's stored LR and weight decay.
   - On step 1, Adam's update is `u = g / (|g| + eps)`, so `max |u|` over a tensor must be just
     under 1 no matter how large the gradient is.
   - If the optimizer applied any LR other than the one stored in the parameter's group, it would
     show up directly: half that LR gives 0.5, double gives 2. Together with check 2 (stored LR =
     multiplier x schedule), this verifies that the intended LR is the one applied.
   - The pass band is [0.99, 1.005].
   - On later steps the RMS of `u` is recorded. On a skipped or zero-LR step, the weights must not
     move at all.
4. **Coordinate check** (reported, not asserted). A fixed probe batch is run in eval mode before
   training and after every step. The RMS change of five activations is recorded: the embedding
   output, block 1, the middle block, the last block, and the logits. The report fits the slope
   of log(change) against log(d_model) across the four hero-like widths; 0 means the change does
   not depend on width.

## Results

### Hard checks: all 10 runs passed

| Run | Groups | Parameters checked | Step-1 max abs(u), lowest param | Step-1 max abs(u), highest param | Highest matrix |
|---|---|---|---|---|---|
| d512 muP | 6 | 207 | 0.9996 | 1.0010 | 1 + 2.5e-5 |
| d1024 muP | 7 | 207 | 0.9993 | 1.0009 | 1 + 4.0e-5 |
| d1536 muP | 8 | 207 | 0.9986 | 1.0009 | 1 + 2.6e-5 |
| d2048 muP | 7 | 207 | 0.9990 | 1.0008 | 1 + 3.3e-5 |
| small muP | 7 | 415 | 0.9974 | 1.0009 | 1 + 4.0e-5 |
| medium muP | 9 | 623 | 0.9945 | 1.0009 | 1 + 3.1e-5 |
| d512 / d1024 / d1536 / d2048 SP | 2 each | 207 each | 0.9958 | 1.0009 | 1 + 2.6e-5 |

Other hard-check results:

- **The LR invariant held:** across all sizes, steps and hidden matrices, the largest value of
  `abs(group LR * fan_in / base_width / scheduled LR - 1)` was **2.53e-8**. That is the rounding
  error from storing the LR as an fp32 tensor (the optimizer is compiled). For example, the
  step-1 LR of 1.6e-4 is stored as 0.00015999999595806003.
- **Logged LRs matched the applied LRs** at every logged step in every run.
- **The step-20 zero-LR step left every weight unmoved.** At step 20 the WSD schedule reaches LR 0;
  in all 10 runs every parameter's fp32 master weight was bit-identical before and after that
  step.
- **No step was skipped** by the loss-spike detector in any run.
- **SP check:** under SP, both groups had multiplier 1. The same step-1 check passed with that
  uniform LR, so the harness measures whichever LRs are applied; it is not tuned to muP's
  values.
- **Expert parallelism was active in the medium run.** Each GPU held routed-expert tensors of
  shape `[64, 3072, 768]` (512 / 8 experts), and those tensors sat in their own expert-parallel
  groups with the expected multipliers (1.333 for `w_up_gate`, 0.6667 for `w_down`).

How to read the step-1 numbers:

- **Values below 1** mostly come from Adam's eps. In very large tensors with small clipped
  gradients, even the largest element of `g / (|g| + eps)` falls a little short of 1. The lowest
  value (0.9945) is a KDA `f_proj_2` in the 42.8B medium hero, where gradients clipped to norm 1
  are spread over the most parameters. The same fp32 rounding described next can also pull a
  value slightly below 1.
- **Values above 1** come from fp32 rounding. Both `W` and the decay factor `1 - lr * wd` are
  fp32, so the recovered `u` carries an error of roughly `ulp(abs(W)) / lr`, which is largest for
  large-valued parameters. The largest step-1 excess over 1, by parameter type:

  | Parameter type | Typical magnitude | Excess under muP | Excess under SP |
  |---|---|---|---|
  | embeddings | std 1.0 (muP), 0.02 (SP) | up to 1.0e-3 | up to 6e-6 |
  | `A_log` | log(1) to log(16), both | up to 9.5e-4 | up to 9.2e-4 |
  | depthwise convs | std 0.5 (muP), 0.02 (SP) | up to 3.9e-4 | up to 2.6e-5 |
  | norm gains, `ssmax_scale` | exactly 1.0, both | up to 3.0e-5 | up to 3.0e-5 |
  | hidden matrices | std about 0.01 to 0.06 (muP), 0.02 (SP) | up to 4.0e-5 | up to 2.6e-5 |

  The excess tracks each parameter's size:
  - Embeddings and convs, which nanochat initializes 50x and 25x larger than SP, overshoot about
    170x and 15x more under muP.
  - `A_log`, initialized the same way under both, overshoots equally under both.
  - Norm gains and `ssmax_scale` all start at exactly 1.0, so every element rounds the same way.
    Their excess is that single rounding error, not a worst case over many different values.

  That pattern is consistent with rounding, not with any LR error. No hidden matrix exceeded 1 by
  more than 4e-5.
- **Sensitivity:** if a parameter had received a different multiplier than intended, the
  step-1 value would move by that ratio. For example, if the medium attention `w_out` had
  received the d_model-based 0.6667 instead of 0.5, it would read as 1.33, far outside the
  observed 0.9945 to 1.0010.

The groups contain several entries with the same multiplier. That is expected: the routed
experts are kept in their own groups by the production group overrides (and are also in the
expert-parallel process group in the medium run), and muP splits each group by multiplier
separately.

### Multipliers by role and size

The muP multiplier (group LR / scheduled LR), with `base_width = 1024`. Roles are prefixed by
block type:

- `attn`: gated full attention plus latent MoE;
- `moe`: KDA plus latent MoE;
- `dense`: block 0, KDA plus the dense 8d MLP.

| Role | fan_in | d512 | d1024 | d1536 | d2048 | small | medium |
|---|---|---|---|---|---|---|---|
| attn `w_q`, `w_k`, `w_v`, `w_g` | d | 2 | 1 | 0.6667 | 0.5 | 1 | 0.6667 |
| attn `w_out` | heads x 128 | 2 | 1 | 0.6667 | 0.5 | 1 | **0.5** |
| KDA `w_q`, `w_k`, `w_v`, `w_b`, `f_proj_1`, `g_proj_1` | d | 2 | 1 | 0.6667 | 0.5 | 1 | 0.6667 |
| KDA `w_out` | heads x 128 x 2 | 1 | 0.5 | 0.3333 | 0.25 | 0.5 | **0.25** |
| KDA `f_proj_2`, `g_proj_2` | 256 (fixed) | 4 | 4 | 4 | 4 | 4 | 4 |
| `latent_down_proj`, router, shared `w_up_gate`, shared `w_down` (MoE blocks) | d | 2 | 1 | 0.6667 | 0.5 | 1 | 0.6667 |
| `latent_up_proj`, routed `w_up_gate` | d/2 | 4 | 2 | 1.333 | 1 | 2 | 1.333 |
| routed `w_down` | d (expert hidden) | 2 | 1 | 0.6667 | 0.5 | 1 | 0.6667 |
| dense `w_up_gate` | d | 2 | 1 | 0.6667 | 0.5 | 1 | 0.6667 |
| dense `w_down` | 8d | 0.25 | 0.125 | 0.0833 | 0.0625 | 0.125 | 0.0833 |
| `lm_head.w_out` | d | 2 | 1 | 0.6667 | 0.5 | 1 | 0.6667 |
| embeddings | - | 1 | 1 | 1 | 1 | 1 | 1 |
| all vector-like params | - | 1 | 1 | 1 | 1 | 1 | 1 |

The real small hero and the d1024 hero-like model have the same d_model and head count. All 41
matrix and embedding roles they share have identical multipliers, as do their vector-like
parameters (all 1).

### How the results match the per-parameter fan_in design

1. **Every multiplier is `base_width / (that parameter's fan_in)`.** Hard check 1 compared every
   parameter in all 10 runs against an expected value computed from its own shape, and found no
   mismatches. The fan_in invariant held to 2.5e-8 for every matrix at every step. Under a single
   model-wide multiplier, that invariant would fail for any matrix whose fan_in is not d_model.
2. **Heads growing faster than d_model lowers exactly the right LRs.** The medium hero and the
   d1536 hero-like model have the same d_model but different head counts (16 versus 12). The
   report lists every role whose multiplier differs between them:

   | Role | fan_in (medium vs d1536) | Multiplier (medium vs d1536) |
   |---|---|---|
   | attn `w_out` | 2048 vs 1536 | 0.5 vs 0.6667 |
   | dense (block 0) KDA `w_out` | 4096 vs 3072 | 0.25 vs 0.3333 |
   | moe KDA `w_out` | 4096 vs 3072 | 0.25 vs 0.3333 |

   These are exactly the matrices that read the head space (24 tensors: 3 attention and 21 KDA),
   and nothing else differs. Every projection from the residual stream got 0.6667 in both,
   including the attention and KDA q/k/v and gate projections, whose output size grew. A single
   model-wide multiplier would have given all of these 0.6667. Going from small to medium, both
   output projections drop 2x (attention `w_out` 1 to 0.5, KDA `w_out` 0.5 to 0.25) while d_model
   grows only 1.5x.
3. **Fixed fan_ins keep a fixed LR.** `f_proj_2` and `g_proj_2` read a 256-wide bottleneck at
   every size and got 4x at every size. A d_model-based multiplier would have scaled them with
   width.
4. **Other fan_ins that are multiples of d_model scale by that multiple.** At every width:
   - KDA `w_out` (fan_in 2d) gets half the multiplier of the d_model projections;
   - `latent_up_proj` and routed `w_up_gate` (fan_in d/2) get double;
   - the dense `w_down` (fan_in 8d) gets 1/8.
5. **The multipliers were actually applied (check 3).** Step-1 updates matched each group's own
   LR to within 0.6% for every parameter (within 0.6% below and 4e-5 above for every matrix).
   That includes the medium hero's 0.5 and 0.25 output
   projections, the 4x low-rank projections and the 0.0625x dense `w_down` at d2048. So the
   per-parameter LRs survive the whole path: `build_groups`, the scheduler (which scales each
   group from its own `initial_lr`), and the fused, compiled `OLMoDDPOptimizer` step, including
   the expert-parallel groups.

### Later steps

For steps 2 to 19, the median over steps of each parameter's RMS(u) (RMS update divided by the
group LR):

- **Hero-like sizes:** 0.23 to 0.38 for every hidden matrix. Across the four widths, the
  per-role values mostly move by about 0.01 (median spread 0.012). The ratio of d2048 to d512 has
  a median of 0.98 and ranges from 0.86 to 1.07. The largest drifts:
  - KDA `f_proj_2` in the KDA + MoE blocks falls from 0.29 to 0.25. This is the fixed-fan_in
    projection, whose LR stays at 4x while its output size grows.
  - The full-attention block's router rises from 0.31 to 0.33.
- **Real small and medium heroes:** 0.15 to 0.47, mostly 0.26 to 0.40. The ends of the range:
  - one KDA `f_proj_2` in the medium hero at 0.15;
  - the routers at up to 0.47. A high RMS(u) means a gradient whose sign is consistent from step
    to step.
- **Vector-like parameters** vary more (0.18 to 0.96). `lm_head.norm.weight` is the high end at
  every size.

These numbers describe Adam's normalized update, not the LR: the LR was verified exactly on step
1, and the group LRs are checked every step. They show no width-dependent breakdown, but they
are not a second LR check. With 20 steps on random data, the small drifts above are not
conclusive either way.

### Coordinate check

Per-coordinate RMS change of the probe activations after 20 steps, across the four hero-like
widths. "Exponent" is the slope of log(change) against log(d_model).

Standard parameterization:

| Activation | d512 | d1024 | d1536 | d2048 | Exponent |
|---|---|---|---|---|---|
| embedding | 0.003113 | 0.003119 | 0.003119 | 0.003118 | +0.00 |
| block 1 | 1.686 | 1.681 | 1.704 | 1.729 | +0.02 |
| middle block | 3.408 | 3.427 | 3.482 | 3.541 | +0.03 |
| last block | 4.718 | 4.748 | 4.808 | 4.872 | +0.02 |
| logits | 0.51 | 0.7198 | 0.8914 | 1.042 | **+0.51** |

muP:

| Activation | d512 | d1024 | d1536 | d2048 | Exponent |
|---|---|---|---|---|---|
| embedding | 0.003631 | 0.003593 | 0.003562 | 0.003534 | -0.02 |
| block 1 | 2.403 | 2.495 | 2.412 | 2.508 | +0.02 |
| middle block | 6.718 | 6.705 | 6.673 | 6.794 | +0.00 |
| last block | 11.65 | 11.45 | 11.41 | 11.42 | -0.02 |
| logits | 1.194 | 1.211 | 1.209 | 1.202 | **+0.01** |

What this signifies:

- **Logits:** under SP the change grows about as sqrt(width), and under muP it is flat. This is
  the expected muP result, and it is consistent with the `lm_head` LR scaling as `1 / d_model`.
- **Hidden activations are flat under both parameterizations.** Every hero block adds
  `RMSNorm(f(x))` to the residual stream (peri-norm), and the embedding is normalized too. Most
  likely this normalizes away width effects inside the trunk. Either way, on this architecture
  the coordinate check only discriminates at the logits. It is consistent with muP but is weak
  evidence for the hidden layers.
- **A model without peri-norm does show the hidden-layer difference.** A separate check on a
  Llama-like model (`src/examples/mup/coord_check.py`: widths 128 to 1024, base width 128, 10
  steps) found:
  - SP: middle-block exponent +2.32, logits +0.55;
  - muP: middle block -0.04, logits -0.02.
- **muP moves the activations more than SP, even at d1024 = base_width.** The muP / SP ratio at
  d1024 is:
  - 1.15 for the embedding;
  - 1.48, 1.96 and 2.41 for blocks 1, middle and last;
  - 1.68 for the logits.

  This is expected. Under fixed-`base_width` anchoring, several multipliers are not 1 at the base
  width, and nanochat init differs from std 0.02. It means the LR of 8e-4 tuned under SP is not a
  muP-tuned LR.
- **The trunk's change grows with depth.** At the same width, more layers means a larger
  last-block change:

  | Model | d_model | Layers | Block 1 | Middle block | Last block | Logits |
  |---|---|---|---|---|---|---|
  | d1024 hero-like | 1024 | 8 | 2.495 | 6.705 | 11.45 | 1.211 |
  | small hero | 1024 | 16 | 1.729 | 7.506 | 17.83 | 1.230 |
  | d1536 hero-like | 1536 | 8 | 2.412 | 6.673 | 11.41 | 1.209 |
  | medium hero | 1536 | 24 | 1.670 | 8.440 | 23.56 | 1.196 |

  The real heroes also differ in expert count (512 versus 64), top-k (16 versus 8), and in the
  medium hero's head count, so this is not a clean depth comparison. muP as designed does not
  scale depth. The logits' change stays at 1.20 to 1.23 in all four models.

## Inconsistencies found during testing

None pointed to a defect in the muP implementation. For completeness:

- **Step-1 values above 1** (up to 1.0010), although Adam's first update is strictly below 1 in
  exact arithmetic. This is fp32 rounding, largest on large-valued parameters, as explained above.
  To allow for it, the pass band was widened from the planned [0.99, 1.0] to [0.99, 1.005].
  Hidden matrices exceeded 1 by at most 4e-5.
- **Harness bug (fixed).** The first smoke run failed a harness check: only 19 of 20 steps had
  logged LR metrics. The final step's metrics are logged after training ends, so the check now
  compares each step's logged LRs as they arrive and tolerates the missing final log. Only rank 0
  had flagged it, so the other ranks hung; failures are now reported after teardown instead of
  raised inside the callback.
- **Plan change.** The LR assertion was planned to run before the optimizer step. The train
  module sets each step's LR inside its optimizer step, so the assertion runs after the step.
- **No learning signal.** On random tokens the cross-entropy stays at about 11.7, which is
  ln(vocab size), under both parameterizations.

## What these tests do not establish

- **LR transfer.** These runs verify mechanics, not training quality. The real test of muP is a
  short LR sweep on real data at two or three sizes: the best LR should stay put under muP and
  shift under SP.
- **The production expert-parallel path.** The medium hero used `sync_1d` expert parallelism,
  not `rowwise_nvshmem`. Optimizer grouping and LRs don't depend on how tokens are moved, but
  that code path was not exercised.
- **Checkpoint resume** with muP optimizer groups (`lr_multiplier`, `initial_lr` and group
  order).
- **Scaling beyond width.** Depth (small to medium also goes from 16 to 24 layers), expert count
  and top-k are not scaled by this muP. Weight decay is not width-scaled either
  (`scale_weight_decay=False`), so the effective decay `lr_l * wd` shrinks with width for every
  matrix whose fan_in grows with width.
- **Non-default options.** These runs used `MuPConfig` defaults. `embedding_lr_multiplier` is
  covered only by unit tests in `src/test/optim/mup_test.py`. `scale_weight_decay=True` was
  later verified with base-model fan-ins on GPU in the scaling-ladders mainline ladder (applied
  per-step decay measured at every rung, plus checkpoint resume); see
  `ladders/mainline/MUP_WEIGHT_DECAY.md` there.
- **Only one `base_width`** (1024) was used.
- **Full independence of the expected values.** The expected multipliers in check 1 were written
  by the same author as the muP code, so both share one reading of the tensor layouts. The
  printed shapes support that reading. For example, routed `w_down` at d512 is `[64, 512, 256]`:
  64 experts, expert hidden 512, latent 256, so fan_in is the hidden size 512. Still, the
  expected values are not a fully independent derivation.

## Other verification

- Unit tests:
  - `src/test/nn/transformer/mup_test.py` (shape registry, and init consistency with the
    registry);
  - `src/test/optim/mup_test.py` (multipliers, group splitting, overrides and the scheduler);
  - `test_build_groups_applies_mup_with_wrapper_prefix` in `src/test/optim/moe_optimizer_test.py`
    (DDP-wrapped parameter names).
- Nanochat init was verified separately on the real hero: every parameter's last writer and std
  are correct, and existing init schemes are unchanged. See `docs/NANOCHAT_INIT_HANDOFF.md`.

## Reproducing

Each hero-like run takes about 7 to 8 minutes on 8 B300s; the small hero takes about 10 and the
medium hero about 11. Results go to `/tmp/mup_hero_check/` (JSON per run, logs in `logs/`). That
directory is not persistent; rerun to regenerate.

```bash
for size in d512 d1024 d1536 d2048; do
  for param in mup sp; do
    PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/mup/hero_lr_check.py run $size $param
  done
done
PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/mup/hero_lr_check.py run small mup
PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/mup/hero_lr_check.py run medium mup
PYTHONPATH=src python src/examples/mup/hero_lr_check.py report
```

`report` exits non-zero if any hard check failed or if the small hero's multipliers differ from
the d1024 hero-like model's.
