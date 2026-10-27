# Handoff: CompletedP (Complete(d)P) parameterization

Written 2026-10-20 against commit `e45401f0c` (branch `tanushy/parameterization`) plus
uncommitted changes. **Nothing has been committed.** Line numbers refer to the working tree at
the time of writing; re-grep before editing.

**Update 2026-10-26: per-matrix base fan-ins.** A matrix's width ratio is now
`r = fan_in / base_fan_in`, its fan-in over the fan-in of the same matrix in the base model,
instead of `fan_in / base_width`. `CompletedPConfig.from_base_model(base_model_config, alpha)`
builds the base model on the meta device and records every block matrix's fan-in in
`CompletedPConfig.base_block_fan_ins`, one mapping per distinct block structure. A block of another
model uses the mapping with exactly its set of matrix names, and building a model raises if none
matches. The LM head's base fan-in is `base_width`, and epsilon still uses
`m_N = d_model / base_width`. So the base model now gets exactly the base init and hyperparameters
(every multiplier 1), and the ratios between two models are unchanged. Configs without
`base_block_fan_ins` keep the old rule and log a warning. The absolute multipliers at the base
quoted below (Section 1.4, item 1; Section 1.6's small-hero values such as KDA `f_proj_2` at 4x
LR; the small-hero group counts) were computed with the old rule.

Companion docs:
- `docs/PARAMETERIZATION_HANDOFF.md`: how init and optimizer groups work in this codebase.
- `docs/NANOCHAT_INIT_HANDOFF.md`: `InitMethod.nanochat`, the spectral init.
- `docs/MUP_HANDOFF.md` and `docs/MUP_VERIFICATION.md`: muP learning rates (`MuPConfig`). CompletedP
  reuses muP's parameter-shape registry and group-splitting code.
- `docs/CompletedP.png`: Table 1 of the paper.

The paper is Mlodozeniec et al., "Completed Hyperparameter Transfer across Modules, Width, Depth,
Batch & Duration" (arXiv:2512.22382). It builds on CompleteP (Dey et al., arXiv:2505.01618). This
doc writes the scheme as "Complete(d)P" when quoting the paper and as "CompletedP" for the code.

Notation (all ratios are "this run / base run"):

| Symbol | Meaning |
|---|---|
| `m_N` | width ratio, `d_model / base_width` |
| `m_L` | depth ratio, `n_layers / base_depth` |
| `m_B` | batch-size ratio, tokens per optimizer step |
| `m_D` | training-duration ratio, total training tokens |
| `alpha` | depth exponent, in `[0.5, 1]` |
| `r` | a weight matrix's own width ratio, `fan_in / base_width` |
| `s` | the SDE factor, `sqrt(m_B / m_D)` |
| `lr`, `eps`, `wd`, `beta1`, `beta2` | the optimizer's base AdamW hyperparameters |
| `std` | the base init std (`init_std`, 0.02 by default) |

---

## 0. Summary

CompletedP is a parameterization: a set of rules that rescale every per-tensor hyperparameter
(init std, learning rate, AdamW epsilon, weight decay, betas) and the residual-branch multipliers
as a function of width, depth, batch size and training duration. Hyperparameters tuned on a small,
short "base" run should then stay (near) optimal on a larger or longer run without retuning.

What was implemented:

- **Model side** (`TransformerConfig.completedp`, a `CompletedPConfig` with `base_width`,
  `base_depth`, `alpha`):
  - every residual branch is scaled by `m_L^-alpha`;
  - the `normal` init of every hidden weight matrix is scaled by `sqrt(1 / r)`. The LM head and
    the in-block matrices with a fixed output size (MoE routers, KDA `f_proj_1` / `g_proj_1`) are
    scaled by `1 / r`.
- **Optimizer side** (`completedp`, a `CompletedPOptimConfig`, on `AdamWConfig`,
  `SkipStepAdamWConfig` or the hero's `OLMoDDPOptimizerConfig`): every parameter group gets its own
  `lr`, `eps`, `weight_decay` and `betas` by role, width, depth, batch size and duration.
- **Supported models**: the generic dense `Transformer` with `default`, `reordered_norm` or
  `peri_norm` blocks, and the hero `OLMoDDPModel` (`moe_fused_v2` blocks with attention, KDA or GDN
  sequence mixers, routed/shared experts and latent MoE). Everything else raises.
- **The real hero**: Section 1.6 shows how the production small and medium hero configs are
  treated, with the small hero as the base. That covers geometry, residual multipliers, every
  parameter's init / lr / eps / weight decay, and the optimizer groups.

```python
from olmo_core.nn.transformer import CompletedPConfig
from olmo_core.optim import CompletedPOptimConfig, OLMoDDPOptimizerConfig

model_config.completedp = CompletedPConfig(base_width=1024, base_depth=16, alpha=1.0)
optim = OLMoDDPOptimizerConfig(
    lr=..., eps=..., betas=..., weight_decay=...,  # base values, tuned at the base run
    completedp=CompletedPOptimConfig(
        base_batch_size=..., batch_size=...,        # tokens per optimizer step
        base_train_tokens=..., train_tokens=...,    # total training tokens
    ),
)
```

Verification (Section 4):

- 86 new unit tests. The broader suites gave 692 passed, 1083 skipped, 40 xfailed and the same 3
  failures as a clean `HEAD` checkout.
- On the hero architecture (real `Trainer`, `OLMoDDPTrainModule`, `OLMoDDPOptimizer`), 32 training
  runs passed every hard check: 24 before the readout change (Section 1.4, item 10) and the 8
  CompletedP runs again after it. For every parameter, the init std, the residual multipliers, and
  the group `lr` / `eps` / `weight_decay` / `betas` match Table 1, the LR schedule keeps every
  group's ratio at every step, and the first AdamW step recomputed from the captured gradients
  matches the observed moments and fp32 weights. So the hyperparameters actually applied, not only
  the stored ones, are the intended ones.
- The exact production small and medium heroes (Section 4.6):
  - Before the readout change they trained on 8 GPUs: the small hero with DDP, the medium hero
    with expert parallelism, `alpha` 1 and 0.5. All hard checks passed.
  - After it, only a 1-GPU host was available. The real configs were rechecked without training:
    the real `init_weights` on CPU and the real optimizer groups. Every parameter's lr, eps,
    weight decay and init std scales from small to medium as Table 1 predicts.
  - The step-1 check on the real configs was not rerun with the new classification; it ran on
    the hero-like models instead (Section 4.4).
- Coordinate checks: on a dense model CompletedP is flat in width and depth while standard
  parameterization (SP) blows up. On the hero it is flat in width. In depth, the hero is flat with
  `alpha=0.5` and shrinks as about `L^-0.36` with `alpha=1` (Section 4.5).
- Throughput: no measurable cost. The residual multiplier is folded into the branch RMSNorm's gains
  on the hero, and more optimizer groups did not slow the optimizer step (Section 4.7).

Not established: that a tuned LR actually transfers (needs an LR sweep on real data), the
production `rowwise_nvshmem` expert-parallel path, pipeline parallelism, FP8, and checkpoint resume
with CompletedP groups.

---

## 1. The Complete(d)P scheme

### 1.1 Base run and ratios

You pick a base run: a model of width `base_width` and depth `base_depth`, trained with batch size
`B_base` tokens per step for `D_base` tokens. You tune `std`, `lr`, `eps`, `wd`, `beta1`, `beta2` on
it. Every other run is described by four ratios relative to the base: `m_N` (width), `m_L` (depth),
`m_B` (batch), `m_D` (duration). Complete(d)P says how each hyperparameter of each tensor scales
with those ratios. At the base run every ratio is 1, and (in the paper's formulation) every rule
reduces to the base values.

### 1.2 Table 1 of the paper, in ASCII

"x f" means "multiply the base value by f". The last column applies to every row of its section.

```text
                                 muP (TP5 Table 3)   Complete(d)P                 all rows also
Forward multipliers
  attention residual             x + Block(x)        x + m_L^-alpha * Block(x)
  MLP residual                   x + Block(x)        x + m_L^-alpha * Block(x)
  unembedding forward            unaugmented         unaugmented

Init variance (x std^2)
  input embedding                x 1                 x 1
  hidden weights                 x m_N^-1            x m_N^-1
  hidden biases / norms          x 1                 x 1
  unembedding norm               x 1                 x 1
  unembedding weights            x m_N^-2            x m_N^-2

Learning rate (x lr)                                                              x sqrt(m_B/m_D)
  input embedding                x 1                 x 1
  hidden weights                 x m_N^-1            x m_N^-1 * m_L^(alpha-1)
  hidden biases / norms          x 1                 x m_L^(alpha-1)
  unembedding norm               x 1                 x 1
  unembedding weights            x m_N^-1            x m_N^-1

AdamW epsilon (x eps)                                                             x sqrt(m_D/m_B)
  hidden weights/biases/norms    x m_N^-1            x m_N^-1 * m_L^-alpha
  QK norms (shared over heads)   n/a                 x m_L^-alpha
  input embedding                x m_N^-1            x m_N^-1
  output weights/biases/norms    x 1                 x 1

Weight decay (x wd)                                                               x sqrt(m_B/m_D)
  hidden weights                 x m_N               x m_N
  unembedding weights            x m_N               x m_N
  everything else                x 1                 x 1

AdamW (1 - beta1), (1 - beta2)                                                    x m_B/m_D
Training iterations              proportional to m_D / m_B
```

Weight decay uses the PyTorch AdamW convention: each step multiplies a weight by
`1 - lr_group * wd_group`.

### 1.3 Where the rules come from

**Width (muP for Adam).** Adam normalizes each gradient entry to roughly +/-1, so a step changes a
matrix with fan-in `N` by about `lr` per entry, and its output by about `lr * N` (the `N` inputs add
up coherently). Keeping that change constant as width grows needs `lr ~ 1/N`. Init variance `1/N`
keeps forward activations of order 1. The unembedding gets init variance `1/N^2` and `lr ~ 1/N` so
the logits' change stays of order 1. CompleteP has a `1/N` forward multiplier on the LM head
instead; Complete(d)P folds it into the init and LR, so fused or cut cross-entropy kernels that
take the raw weight still work. Weight decay `x m_N` keeps `lr * wd` constant, because PyTorch
couples them. Epsilon tracks the gradient scale: hidden gradients shrink as `1/N`, so does `eps`.
The embedding's gradient also scales as `1/N` (the paper fixes CompleteP here). A gain shared
across attention heads sums its gradient over the heads, which cancels the `1/N`, so its epsilon
does not scale with width (the paper's "QK norms" row, Appendix B.1).

**Depth (CompleteP / depth-muP).** Each residual branch is multiplied by `m_L^-alpha`.
- With `alpha = 1` (CompleteP), each block contributes `1/L` of the total update, like steps of an
  ODE.
- With `alpha = 1/2` (depth-muP), each block contributes `1/sqrt(L)`, like an SDE, which keeps more
  diversity between layers.

The paper reports transfer for all `alpha` in `[1/2, 1]`, slightly better at its largest size with
`alpha = 1/2`. The hidden LR scales as `m_L^(alpha-1)` so a block's update, times the branch
multiplier, has the right size. The hidden epsilon scales as `m_L^-alpha` because the branch
multiplier scales every gradient inside the block by `m_L^-alpha`.

**Batch size and duration (SDE rules).** AdamW is treated as a discretization of a stochastic
differential equation.
- Multiplying the batch size by `k` reduces the gradient noise standard deviation by `sqrt(k)`.
  Keeping the same SDE needs `lr x sqrt(k)`, `wd x sqrt(k)`, `eps / sqrt(k)`, `(1 - beta) x k`, and
  `k` times fewer steps.
- For duration, the paper finds the best LR falls as `1/sqrt(k)` when training `k` times longer at
  a fixed batch size. It treats more tokens like a better gradient signal-to-noise ratio, which
  gives the inverse factors: `lr / sqrt(k)`, `wd / sqrt(k)`, `eps x sqrt(k)`, `(1 - beta) / k`.

Together the factors depend only on `m_B / m_D`, which equals `base_steps / steps`.

**Per-module hyperparameters.** The paper's second contribution is tuning separate multipliers per
module type (and per depth) at the base scale. The parameterization above transfers them. This
implementation has no dedicated feature for that. `group_overrides` already provide per-module base
values for `lr`, `eps`, `weight_decay` and `betas`, and CompletedP scales those (Section 3.6).
Per-module init scales and residual multipliers are not supported.

### 1.4 How this implementation reads the table

Decisions settled with the user before implementing:

1. **Each matrix uses its own width ratio** for init, LR and weight decay, instead of one
   model-wide `m_N`. This matters on the hero, whose dimensions do not grow together.
   - Small -> medium: `d_model` x1.5, attention head space x2, KDA's 256-wide low-rank projections
     x1.
   - Originally `r = fan_in / base_width`, as in `MuPConfig`. Then not every multiplier was 1 even
     at the base run: KDA `f_proj_2` got 4x LR and 2x init std at `base_width = 1024`, for
     example.
   - Now `r = fan_in / base_fan_in`, the same matrix's fan-in in the base model (see the update
     note at the top), so every multiplier is 1 at the base run.
2. **The base init std is `init_std`.** CompletedP scales the existing `normal` init (truncated
   normal at +/-3 std) by the width factors. At `fan_in == base_width` the init is bit-identical to
   today's hero init. Setting `init_std = base_width ** -0.5` gives the hidden matrices a
   `1/sqrt(fan_in)` init.

Decisions made while implementing:

3. **Epsilon uses the residual-stream ratio `m_N = d_model / base_width`**, not `r`. The gradient
   scale that epsilon tracks is set by the residual stream's width (the LM head's fan-in), not by
   each matrix's fan-in.
4. **The embedding norm** (`embedding_norm`, which the hero applies right after the embeddings) is
   treated as part of the input layer: LR x1, epsilon `x 1/m_N`, weight decay x1.
5. **Head-shared gains** get the "QK norms" epsilon rule (`x m_L^-alpha`):
   - head-wise QK-norm gains with one `[head_dim]` gain (`use_head_qk_norm=True`,
     `qk_norm_per_head_gains=False`);
   - the KDA/GDN output norm `o_norm` (`[head_v_dim]`, shared across heads). This is an extension of
     the paper's argument, not in the paper.

   Per-head QK gains (`[n_heads, head_dim]`) and full-width QK norms are not shared, so they follow
   the ordinary hidden-vector rule.
6. **MoE weights** (routed and shared experts, latent projections) are hidden matrices, with
   fan-in from module attributes as in muP (Section 2.3). Routers are readout matrices (next
   item).
10. **In-block matrices with a fixed output size are readout matrices.** These are the MoE
    routers (one output per expert; the expert count does not grow with width) and KDA's
    `f_proj_1` / `g_proj_1` (they project into the fixed 256-wide bottlenecks of the decay and
    output gates).
    - By muP's definition (which of a matrix's dimensions grow with width) they are output-like
      rather than hidden. They get the output-weight width rules: init std `x 1/r`, LR `/ r`,
      epsilon not width-scaled, weight decay `x r`.
    - Because they sit inside the residual blocks, they keep the hidden depth rules: LR
      `x m_L^(alpha-1)`, epsilon `x m_L^-alpha`.
    - Compared with the hidden rule, only the init and the epsilon change. This was settled with
      the user after the 8-GPU runs. Afterwards the hero-like training checks were rerun (Section
      4.4) and the real configs were rechecked without training (Section 4.6).
    - Matrices whose input size is fixed (KDA `f_proj_2` / `g_proj_2`) stay hidden matrices. Their
      rules depend only on fan-in, so the input-like reading gives the same result.
7. **Depthwise convolution taps, `A_log`, `dt_bias`, `ssmax_scale`, attention sinks and biases**
   are hidden vectors. Their init is unchanged.
8. **The unembedding norm's epsilon follows the table literally (x1).** One can argue its gradient
   scales as `1/N`; this was not changed.
9. **Peri-norm blocks** (`x + RMSNorm(f(RMSNorm(x)))`, the hero) apply the multiplier to the whole
   branch, outside the post-branch norm. It is folded into that norm's gains (Section 2.3).

### 1.5 Per-parameter rules as implemented

Multipliers on the base values. `s = sqrt(m_B / m_D)`, `r = fan_in / base_width`,
`m_N = d_model / base_width`, `m_L = n_layers / base_depth`.

| Role (`CompletedPRole`) | Examples | init std | lr | eps | weight decay |
|---|---|---|---|---|---|
| `input_embedding` | `embeddings.weight` | unchanged (`embedding_init_std`, else `init_std`) | `s` | `1/(m_N s)` | `s` |
| `input_vector` | `embedding_norm.weight` | unchanged | `s` | `1/(m_N s)` | `s` |
| `hidden_matrix` | every other linear and expert weight in `blocks.*` | `x sqrt(1/r)` | `m_L^(alpha-1) s / r` | `m_L^-alpha / (m_N s)` | `r s` |
| `hidden_readout_matrix` | in-block matrices with a fixed output size: MoE routers, KDA `f_proj_1`, `g_proj_1` | `x 1/r` | `m_L^(alpha-1) s / r` | `m_L^-alpha / s` | `r s` |
| `hidden_vector` | norm gains, biases, `A_log`, `dt_bias`, convs, per-head QK gains | unchanged | `m_L^(alpha-1) s` | `m_L^-alpha / (m_N s)` | `s` |
| `hidden_head_shared_vector` | shared head-wise QK-norm gains, KDA/GDN `o_norm` | unchanged | `m_L^(alpha-1) s` | `m_L^-alpha / s` | `s` |
| `output_matrix` | `lm_head.w_out.weight` | `x 1/r` | `s / r` | `1/s` | `r s` |
| `output_vector` | `lm_head.norm.weight`, LM-head bias | unchanged | `s` | `1/s` | `s` |

Every block's two residual branches are multiplied by `m_L^-alpha`. Both betas become
`1 - (1 - beta) * m_B / m_D` in every group.

### 1.6 The real hero: small as the base, medium as the target

This section shows how CompletedP treats the production small and medium hero configs
(`build_model_config("small" | "medium")` in `src/examples/olmo_ddp/olmoe3_small_medium_models.py`)
when the small hero is the base. Everything below was computed with the library itself
(`completedp_multipliers`, `named_completedp_params`, `OLMoDDPOptimizerConfig.build_groups`) on
both configs built on the meta device. It uses the per-head QK-norm gains that
`olmoe3_small_integration.py` turns on and `use_cute_kernel=False`; neither changes the
multipliers.

**Geometry.** The dimensions grow by different factors, which is why each matrix uses its own
width ratio `r`.

| | small | medium | growth |
|---|---|---|---|
| `d_model` | 1024 | 1536 | x1.5 (`m_N = 1.5`) |
| blocks | 16 | 24 | x1.5 (`m_L = 1.5`) |
| heads (KV heads), head dim | 8 (4), 128 | 16 (8), 128 | heads x2 |
| attention head space = attention `w_out` fan-in | 1024 | 2048 | x2 |
| KDA value dim = KDA `w_out` fan-in | 2048 | 4096 | x2 |
| KDA low-rank dim = `f_proj_2`, `g_proj_2` fan-in | 256 | 256 | x1 |
| latent dim = `latent_up_proj`, routed `w_up_gate` fan-in | 512 | 768 | x1.5 |
| routed-expert and MoE-block shared-expert hidden size | 1024 | 1536 | x1.5 |
| block 0 dense MLP hidden size = its `w_down` fan-in | 8192 | 12288 | x1.5 |
| routed experts, top-k | 512, 16 | 512, 16 | x1 |
| full-attention blocks | 7, 15 | 7, 15, 23 | |
| parameters, total / active | 12.50B / 794M | 42.76B / 2.39B | |
| parameter tensors | 415 | 623 | |

**Configuration.** `base_width = 1024`, `base_depth = 16` (the small hero's shape), `std = 0.02`,
and the same batch size and duration for both runs (batch/duration effects are covered below):

```python
CompletedPConfig(base_width=1024, base_depth=16, alpha=1.0)   # or alpha=0.5, on both models
```

**Forward pass.** In the medium hero, all 48 residual branches (2 per block, 24 blocks) are
multiplied by `1.5^-alpha`: 0.667 for `alpha = 1`, 0.816 for `alpha = 0.5`. Every hero block is
peri-norm with RMSNorm, so the multiplier is folded into each block's `attention_norm` and
`feed_forward_norm` gains (`output_scale`), and the residual adds stay plain additions. In the
small hero the multiplier is 1 and nothing changes. The LM head forward is unchanged.
`embed_scale = sqrt(d_model)` (32 vs 39.2) has no effect, because the embedding norm follows it.

**Per-parameter treatment.** Each cell is `lr / eps / wd` multiplier, then init std, all relative
to the base values.

| Parameter (fan-in small -> medium) | small (the base) | medium, alpha = 1 | medium, alpha = 0.5 |
|---|---|---|---|
| residual multiplier | 1 | 0.667 | 0.816 |
| hidden matrices with fan-in `d_model` (1024 -> 1536): KDA `w_q`, `w_k`, `w_v`, `w_b`; attention `w_q`, `w_k`, `w_v`, `w_g`; `latent_down_proj`; shared `w_up_gate`; MoE-block shared `w_down`; routed `w_down` | 1 / 1 / 1, 0.02 | 0.667 / 0.444 / 1.5, 0.0163 | 0.544 / 0.544 / 1.5, 0.0163 |
| readout matrices, fan-in `d_model` (1024 -> 1536), output size fixed: routed-experts router (512 experts), KDA `f_proj_1`, `g_proj_1` (256) | 1 / 1 / 1, 0.02 | 0.667 / 0.667 / 1.5, 0.0133 | 0.544 / 0.816 / 1.5, 0.0133 |
| attention `w_out` (1024 -> 2048) | 1 / 1 / 1, 0.02 | 0.5 / 0.444 / 2, 0.0141 | 0.408 / 0.544 / 2, 0.0141 |
| KDA `w_out` (2048 -> 4096) | 0.5 / 1 / 2, 0.0141 | 0.25 / 0.444 / 4, 0.01 | 0.204 / 0.544 / 4, 0.01 |
| KDA `f_proj_2`, `g_proj_2` (256 -> 256) | 4 / 1 / 0.25, 0.04 | 4 / 0.444 / 0.25, 0.04 | 3.27 / 0.544 / 0.25, 0.04 |
| `latent_up_proj`, routed `w_up_gate` (512 -> 768) | 2 / 1 / 0.5, 0.0283 | 1.33 / 0.444 / 0.75, 0.0231 | 1.09 / 0.544 / 0.75, 0.0231 |
| block 0 dense shared `w_down` (8192 -> 12288) | 0.125 / 1 / 8, 0.0071 | 0.083 / 0.444 / 12, 0.0058 | 0.068 / 0.544 / 12, 0.0058 |
| hidden vectors: norm gains, per-head QK-norm gains, `A_log`, `dt_bias`, `ssmax_scale`, convs, biases | 1 / 1 / 1, init unchanged | 1 / 0.444 / 1 | 0.816 / 0.544 / 1 |
| KDA `o_norm` (shared across heads) | 1 / 1 / 1, init unchanged | 1 / 0.667 / 1 | 0.816 / 0.816 / 1 |
| `embeddings` | 1 / 1 / 1, 0.02 | 1 / 0.667 / 1, 0.02 | same as alpha = 1 |
| `embedding_norm` | 1 / 1 / 1, init unchanged | 1 / 0.667 / 1 | same as alpha = 1 |
| `lm_head.w_out` (1024 -> 1536) | 1 / 1 / 1, 0.02 | 0.667 / 1 / 1.5, 0.0133 | same as alpha = 1 |
| `lm_head.norm` | 1 / 1 / 1, init unchanged | 1 / 1 / 1 | 1 / 1 / 1 |

How to read the table:
- **Head space versus `d_model`.** Matrices that read the head space (attention and KDA `w_out`)
  get half their small LR in the medium hero (1 -> 0.5 and 0.5 -> 0.25), more than the x0.667
  that `d_model`'s growth alone would give. The head space doubles while `d_model` grows 1.5x.
- **Fixed dimensions.** KDA `f_proj_2` / `g_proj_2` read a fixed 256-wide bottleneck, so their
  init, LR and weight decay do not change. Only their epsilon (set by the residual stream) and the
  depth factors do.
- **Readout matrices.** The router, `f_proj_1` and `g_proj_1` read the residual stream but have a
  fixed output size. They get the same LR and weight decay as the hidden matrices with fan-in
  `d_model`, but two things differ:
  - Their init std falls as `1/r` (0.02 -> 0.0133) rather than `sqrt(1/r)` (0.02 -> 0.0163).
  - Their epsilon drops only by the depth factor (x0.667 or x0.816), not by the width factor too.
- **Epsilon.** Every hidden epsilon falls by `m_N * m_L^alpha` (2.25 for `alpha = 1`). The
  shared-gain `o_norm` drops only the depth part. The embeddings and the embedding norm drop only
  the width part. The LM head keeps the base epsilon.
- **QK norms.** Per-head QK gains, as in the integration script, are hidden vectors (epsilon
  x0.444 at `alpha = 1`). The plain `build_model_config` uses gains shared across heads; those
  would be head-shared like `o_norm` (epsilon x0.667).
- **Weight decay** scales with each matrix's `r`, so `lr * wd` stays at its base value at
  `alpha = 1`. At `alpha = 0.5` it falls by `1.5^-0.5` inside the blocks.

**Optimizer groups.** With the production overrides (embeddings at weight decay 0, written as
`*embeddings.weight` so it matches under DDP, and the routed-expert group):
- **Small: 8 groups.** One holds all 326 tensors whose multipliers are all 1: every matrix with
  fan-in 1024 (the readout matrices included), all vectors, `embedding_norm` and the LM head. The
  others are KDA `f_proj_2`/`g_proj_2`, KDA `w_out`, the dense `w_down`, `latent_up_proj`, the
  embeddings, and the two routed-expert groups.
- **Medium: 14 groups at `alpha = 1`, 15 at `alpha = 0.5`.** The 65 readout matrices (23 routers,
  21 KDA blocks x 2) form a group of their own. At `alpha = 0.5`, `o_norm` and `embedding_norm`
  also no longer share an LR.
- **Under expert parallelism** (the medium hero runs with EP), the two routed-expert groups land in
  the expert-parallel partition and the rest in the dense one; the multipliers and the total are
  unchanged.
- Standard parameterization has 2 groups today, or 3 once the embedding pattern matches.

**What changes for the small hero itself.** At the base run every ratio is 1:
- the residual multipliers are 1, every epsilon equals the base epsilon, and the betas are the base
  betas;
- every matrix with fan-in 1024 (hidden or readout, and the LM head, embeddings and all vectors)
  gets exactly today's init (std 0.02, truncated at +/-3 std), LR and weight decay;
- only the matrices whose fan-in differs from 1024 change. KDA `w_out` gets std 0.0141, LR x0.5,
  wd x2. KDA `f_proj_2` / `g_proj_2` get std 0.04, LR x4, wd x0.25. `latent_up_proj` and routed
  `w_up_gate` get std 0.0283, LR x2, wd x0.5. The block 0 dense `w_down` gets std 0.0071, LR
  x0.125, wd x8.

So the production LR (1.1e-3 for the hero, 1.85e-3 for the integration runs) is a reasonable
starting point. It is not a CompletedP-tuned value; retune the base hyperparameters on the small
hero with CompletedP enabled before transferring to the medium hero. Today's production also trains
embeddings with weight decay 0.1, because the `embeddings.weight` override does not match under DDP
(`docs/PARAMETERIZATION_HANDOFF.md` Section 2.6). Decide which one the base run should use.

Section 4.6 confirms every value in this section, the group counts and the folded residual
multipliers on both configs. With the readout change this was checked on the real init and
optimizer groups without training; the 8-GPU training runs predate it.

**Batch size and duration.** If the medium run also changes the batch size or token count
relative to the small base run, every group additionally gets the SDE factors from Section 1.5.
- Twice the base batch size with the same number of tokens (`m_B = 2`, `m_D = 1`, `s = 1.414`):
  every LR and weight decay x1.414, every epsilon /1.414, and betas `(0.9, 0.95)` become
  `(0.8, 0.9)`.
- Twice as many tokens at the base batch size (`m_D = 2`): LR and weight decay x0.707, epsilon
  x1.414, and betas `(0.95, 0.975)`.
- Both doubled (the same number of steps): no change.

---

## 2. How it is integrated into the codebase

### 2.1 Data flow

```text
TransformerConfig.completedp = CompletedPConfig(base_width, base_depth, alpha)
  |
  | build()
  v
Transformer.__init__ / OLMoDDPModel.__init__
  - completedp_block_configs(): copy every resolved block config with
      attention_residual_alpha = feed_forward_residual_alpha = m_L^-alpha
  - build blocks (dense: ResidualStream(alpha); DDP block: fold alpha into the branch norm)
  - validate_completedp_model(); store model.completedp
  |
  | init_weights()
  v
normal init (unchanged) -> apply_completedp_init_(): scale matrices in place
  |
  | optimizer build (train module)
  v
optim.completedp = CompletedPOptimConfig(batch / token counts)
  build_groups(model)
    1. expand group_overrides, add the default group               (existing code)
    2. apply_completedp():
         find_completedp_model(model) -> CompletedPConfig, d_model, n_layers
         named_completedp_params(model) -> {name: (role, fan_in)}
         completedp_multipliers() -> {name: {lr, eps, weight_decay}}
         split_groups_by_multipliers(): one sub-group per distinct multiplier triple,
           opts[k] = base_k * m_k, opts["<k>_multiplier"] = m_k
         completedp_betas(): set "betas" on every group
  construct the optimizer; set "initial_lr" = "lr" per group      (existing code)
  |
  | every step (existing code)
  v
Scheduler.set_lr(group): lr = schedule(group["initial_lr"], step)  -> ratios preserved
optimizer step reads each group's lr, betas, eps, weight_decay
```

The model carries the width/depth settings; the optimizer reads them from the model. So the
residual multiplier and the depth-scaled LR cannot disagree. The batch/duration settings live on
the optimizer config because they describe the training run, not the model.

### 2.2 Files

New:

| File | Contents |
|---|---|
| `src/olmo_core/nn/transformer/completedp.py` | Model side: config, roles, validation, init scaling. |
| `src/olmo_core/optim/completedp.py` | Optimizer side: config, multipliers, betas, group splitting. |
| `src/test/nn/transformer/completedp_test.py` | 49 model-side tests. |
| `src/test/optim/completedp_test.py` | 24 optimizer-side tests. |
| `src/examples/completedp/coord_check.py` | Dense width/depth coordinate check (1 GPU, about 20 s). |
| `src/examples/completedp/hero_check.py` | Hero-architecture hard checks, coordinate check and throughput A/B; also trains the production small/medium heroes on all GPUs and reports their scaling against Table 1. |

Modified:

| File | Change |
|---|---|
| `src/olmo_core/nn/transformer/config.py` | Field `TransformerConfig.completedp` (line 349). `build()` raises for types other than `default` / `moe_fused_v2` (389) and forwards it (414; `OLMoDDPModelConfig.build` at 2105). |
| `src/olmo_core/nn/transformer/model.py` | `Transformer.__init__(completedp=...)` (130): sets the residual multipliers (158), stores and validates (185). `init_weights` scales the init (450). |
| `src/olmo_core/nn/ddp/model.py` | `OLMoDDPModel.init_weights` scales the init (985). |
| `src/olmo_core/nn/ddp/block.py` | `OLMoDDPTransformerBlock` accepts `attention_residual_alpha` / `feed_forward_residual_alpha` (previously rejected; 472), folds them into the branch norms (593, `_fold_into_norm` 1514) and applies any remainder in `_res_norm_attn` / `_res_norm_mlp` (1526-1549). |
| `src/olmo_core/nn/layer_norm.py` | `LayerNorm.output_scale` (220), `supports_output_scale()` (233), `_affine()` (242), used by `LayerNorm.forward` and `RMSNorm.forward`. |
| `src/olmo_core/optim/mup.py` | New generic `split_groups_by_multipliers` (109); `split_groups_by_lr_multiplier` (158) is now a wrapper with unchanged output. |
| `src/olmo_core/optim/config.py` | Field `OptimConfig.completedp` (94), class flag `_supports_completedp` (101), `_apply_completedp` (107) called in `build_groups` (199), popped in `build` (229). `MatrixAwareOptimConfig.build` raises (398). |
| `src/olmo_core/optim/adamw.py` | `_supports_completedp = True` on `AdamWConfig` (247) and `SkipStepAdamWConfig` (291). |
| `src/olmo_core/optim/moe_optimizer.py` | Field `OLMoDDPOptimizerConfig.completedp` (159), applied in `build_groups` (294), popped in `build` (383). |
| `src/olmo_core/nn/transformer/__init__.py`, `src/olmo_core/optim/__init__.py` | Export `CompletedPConfig`, `CompletedPOptimConfig`. |
| `src/test/nn/ddp/block_test.py`, `src/test/nn/layer_norm_test.py`, `src/test/optim/moe_optimizer_test.py` | 13 new tests (Section 4.1). |
| `CHANGELOG.md` | One `### Added` entry under `## Unreleased`. |

Not changed: `nn/transformer/init.py` (the init schemes), the schedulers, the train modules, and
every hero script.

### 2.3 Model side

**`CompletedPConfig`** (`nn/transformer/completedp.py:37`). Fields `base_width`, `base_depth`,
`alpha = 1.0`. `__post_init__` rejects non-positive bases and `alpha` outside `[0.5, 1]`. Helpers:
`width_ratio(fan_in)`, `depth_ratio(n_layers)`, `residual_multiplier(n_layers)`.

**Residual multipliers.** `completedp_block_configs` (161) runs in `Transformer.__init__` on the
resolved per-layer block configs (so `block`, `block_overrides` and `block_pattern` are all
covered).
- It copies each config with `dataclasses.replace(..., attention_residual_alpha=c,
  feed_forward_residual_alpha=c)`; the user's config is not modified.
- It raises if a block type is not `default`, `reordered_norm`, `peri_norm` or `moe_fused_v2`, if
  the config is a `NemotronBlockConfig`, or if the user already set a residual alpha.
- Dense blocks already supported these fields through `ResidualStream`
  (`torch.add(residual, branch, alpha=c)`).
- `OLMoDDPTransformerBlock` used to reject them. It now stores them (`None` means 1.0), and when the
  block is not pre-norm, each branch ends in a norm, so it folds the multiplier into that norm:
  `x + c * (g * n) == x + (c * g) * n`, where `g` is the gain and `n` the normalized branch.
- `_fold_into_norm` sets `norm.output_scale = c` and leaves an add multiplier of 1. The add then
  stays the plain `residual + branch` that runs today.
- If the norm cannot fold (pre-norm blocks, norms without an affine weight, or norm classes that
  override `forward`), the block uses `torch.add(..., alpha=c)` instead.
- Every expert-parallel and two-batch-overlap path goes through `_res_norm_attn` / `_res_norm_mlp`,
  so these two helpers are the only residual adds.

**`LayerNorm.output_scale`** (`nn/layer_norm.py:220`) is a plain attribute (not a parameter or
buffer, so not in the state dict), default 1.0. `_affine()` returns `param.type_as(x)` when it is
1.0 (the old code, bit-identical), and otherwise `param.type_as(x) * output_scale`. The scale is
applied after casting to the compute dtype, so bf16 gains are not re-rounded. It scales the bias
too. `supports_output_scale()` is true only for `LayerNorm` and `RMSNorm` with an affine weight.
Mathematically the fold is exact: the forward and the gain's gradient are the same as an explicit
multiplier.

**Validation** (`validate_completedp_model`, 207), at the end of `Transformer.__init__`:
- `init_method` must be `InitMethod.normal`, since CompletedP rescales it;
- no tied embeddings, since the input and output rules differ;
- the LM head must be exactly `LMHead`, not `NormalizedLMHead`;
- every block must be exactly `TransformerBlock`, `ReorderedNormTransformerBlock`,
  `PeriNormTransformerBlock` or `OLMoDDPTransformerBlock`;
- every sequence mixer must be exactly `Attention`, `FusedAttentionV2`, `KimiDeltaAttention` or
  `GatedDeltaNet`.

It warns if `embed_scale` is set without an `embedding_norm`, because a width-dependent embedding
multiplier breaks the input-embedding rule (with the hero's embedding norm it has no effect).

**Init** (`apply_completedp_init_`, 361), called at the end of `Transformer.init_weights` and
`OLMoDDPModel.init_weights`, after the `normal` init:
- under `no_grad` it multiplies each hidden matrix by `sqrt(base_width / fan_in)`, and
  `lm_head.w_out` and each readout matrix by `base_width / fan_in`; everything else is untouched;
- scaling a truncated normal keeps its +/-3 std truncation;
- it draws no new random numbers, so it does not disturb the RNG streams of the existing scheme;
- it is an element-wise multiply, so it applies unchanged to expert-parallel shards (exercised by
  the medium hero run, Section 4.6) and to DTensors (not exercised);
- the DDP train module refreshes rowwise-FP8 caches after `init_weights`, so they see the scaled
  weights.

This relies on `normal` initializing every matrix of the supported modules with `init_std`; a
unit test guards that precondition (Section 4.1).

**Roles** (`named_completedp_params`, 303). Built on muP's `named_param_shapes`
(`nn/transformer/mup.py`), which reads fan-ins from module attributes rather than tensor shapes:
- routed `w_up_gate` is `[E, 2H, D]`, `w_down` is `[E, H, D]`, shared `w_up_gate` is `[D, E*2H]`,
  and the router weight is flattened 1-D;
- under latent MoE, the routed experts' `d_model` is the latent width.

Location comes from finding the `Transformer` inside the model (tolerating a `module.` wrapper
prefix) and checking whether a parameter lives under `embeddings`, `embedding_norm`, `blocks` or
`lm_head`. Head-shared vectors come from `_head_shared_param_names`. Readout matrices come from
`_readout_param_names`, which picks the weight of every `MoERouterV2` / `MoERouter` and KDA's
`f_proj_1` / `g_proj_1`. Any parameter it cannot place raises.

### 2.4 Optimizer side

**`CompletedPOptimConfig`** (`optim/completedp.py:32`). Fields `base_batch_size`, `batch_size`,
`base_train_tokens`, `train_tokens`, all in tokens and all optional. Each pair must be both set or
both unset (unset means a ratio of 1). They are explicit because the optimizer is built before the
trainer and data loader exist. The properties `batch_size_ratio` and `duration_ratio` give `m_B`
and `m_D`.

**Multipliers** (`completedp_multipliers`, 112). Finds the model's `CompletedPConfig`, `d_model` and
`n_layers` (`find_completedp_model`), classifies every parameter, and returns
`{name: {"lr": ..., "eps": ..., "weight_decay": ...}}` per Section 1.5. It raises if the model was
not built with CompletedP.

**Betas** (`completedp_betas`, 164). Computes `1 - (1 - beta) * m_B / m_D` for both betas. They are
returned unchanged when the ratio is 1, and it raises if a result leaves `[0, 1)`.

**Group splitting** (`split_groups_by_multipliers`, `optim/mup.py:109`). This generalizes muP's
splitter.
- Each group (an override or the default group) is split into one sub-group per distinct multiplier
  triple, in order of first appearance.
- Each sub-group gets `opts[k] = base_k * m_k`, where `base_k` is the group's own value if the
  override sets one, else the optimizer's. It also records `"<k>_multiplier"`.
- Other options are kept. Groups stay disjoint.
- `split_groups_by_lr_multiplier` (muP) is now a thin wrapper; its output is unchanged and the muP
  tests pass unmodified.

**Wiring** (`apply_completedp`, 189). It runs the checks, merges multipliers across model parts,
splits the groups and sets `betas` on every group.
- It raises if the model uses CompletedP and the optimizer doesn't (or the reverse), if the
  optimizer is not AdamW, or if `mup` is also set.
- Generic path: `OptimConfig._apply_completedp` in `build_groups`. Only `AdamWConfig` and
  `SkipStepAdamWConfig` set `_supports_completedp`. Muon and Dion (`MatrixAwareOptimConfig`) raise
  in `build`.
- Hero path: `OLMoDDPOptimizerConfig.build_groups`, after the default group is inserted first. It
  runs for both the dense (`pg="dp"`) and expert-parallel (`pg="ep_dp"`) partitions, because
  `build()` calls `build_groups` once per partition.

Group order is unchanged: the generic path puts overrides first and the default group last; the DDP
path puts the default group first. Each group then expands into its sub-groups.

Properties that make this cheap and safe:
- **Logging.** `OLMoDDPOptimizerConfig.build` logs every group's fields and parameter names, so
  `lr`, `lr_multiplier`, `eps`, `eps_multiplier`, `weight_decay`, `weight_decay_multiplier` and
  `betas` appear in the training log automatically.
- **Compile.** With `compile=True`, `lr` is a tensor and `eps`, `betas`, `weight_decay` are Python
  floats inside the compiled `_step_foreach`. They never change during training, so it compiles
  once (a slightly larger graph; no recompiles).
- **Communication.** The DDP flat sync buffers are keyed by partition, not by group, so more groups
  do not change communication.

### 2.5 Relation to muP and nanochat

CompletedP does not use `InitMethod.nanochat` or `MuPConfig`.
- **Init.** It scales the `normal` init, with LM-head std `~ 1/d` (variance `1/N^2`, as in
  TP5/Complete(d)P). Nanochat's std for the LM head is `~ 1/sqrt(d)`, and its hidden matrices have
  an extra `min(1, sqrt(fan_out / fan_in))` factor that the paper does not have.
- **LR.** At the base depth, batch size and duration, CompletedP's learning rates equal muP's
  `base_width / fan_in` (a unit test checks this).
- **Combining.** Setting both `mup` and `completedp` raises.

---

## 3. Using CompletedP in a training run

### 3.1 Workflow

1. Pick the base run (width, depth, batch size, token count) and set the four base values in the
   configs. They must stay identical across every run you want the hyperparameters to transfer
   between. Setting `base_width` to each model's own `d_model` turns the width scaling off.
2. Tune `init_std`, `lr`, `eps`, `weight_decay`, `betas` (and optionally per-module overrides,
   Section 3.6) on the base run, **with CompletedP enabled**. At the base run CompletedP is not
   exactly the standard setup (per-matrix `r` factors, Section 1.4), so values tuned without it do
   not carry over.
3. For the target run, change only the model shape, batch size and training tokens; keep every base
   value.
4. Choose `alpha`. The paper reports transfer for `0.5 <= alpha <= 1`. On the hero, `alpha = 0.5`
   gave a flat depth profile in the coordinate check while `alpha = 1` did not (Section 4.5).
   Decide with a short LR sweep at two depths before committing (Section 5).

### 3.2 Model config

```python
from olmo_core.nn.transformer import CompletedPConfig, InitMethod

model_config.init_method = InitMethod.normal       # required (the default)
model_config.init_std = 0.02                       # base init std
model_config.tie_word_embeddings = False           # required (the default)
model_config.completedp = CompletedPConfig(base_width=1024, base_depth=16, alpha=1.0)
```

Leave `attention_residual_alpha` / `feed_forward_residual_alpha` unset on the block configs;
CompletedP sets them and raises if they are already set.

### 3.3 Optimizer config

```python
from olmo_core.optim import CompletedPOptimConfig

optim_config.lr = ...             # base values, read when the optimizer is built,
optim_config.eps = ...            # so late changes (config.optim.lr = LR) are honored
optim_config.weight_decay = ...
optim_config.betas = (...)
optim_config.completedp = CompletedPOptimConfig(
    base_batch_size=BASE_BATCH_TOKENS, batch_size=BATCH_TOKENS,
    base_train_tokens=BASE_TOTAL_TOKENS, train_tokens=TOTAL_TOKENS,
)
```

`optim_config` is an `AdamWConfig`, a `SkipStepAdamWConfig` or an `OLMoDDPOptimizerConfig`. Pass
`CompletedPOptimConfig()` with no arguments to scale only width and depth. The batch size and token
count are not read from the trainer: keep them in sync with `global_batch_size` and `max_duration`
yourself.

### 3.4 Example: a hero training script

The hero scripts (e.g. `src/examples/olmo_ddp/olmoe3_small_integration.py`) build their configs in
`model_config(common)` and `train_module_config(common)` and define `BATCH` (tokens per step) and
`TOKENS`. A CompletedP variant would add:

```python
from olmo_core.nn.transformer import CompletedPConfig
from olmo_core.optim import CompletedPOptimConfig, OptimGroupOverride

COMPLETEDP = CompletedPConfig(base_width=1024, base_depth=16, alpha=0.5)  # alpha: Section 3.1
BASE_BATCH = 16_777_216            # the base run's tokens per step
BASE_TOKENS = 6000 * BASE_BATCH    # the base run's training tokens


def model_config(common):
    model = ...                    # existing code
    model.completedp = COMPLETEDP
    return model


def train_module_config(common):
    config = ...                   # existing code
    config.optim.lr = LR           # the base LR, tuned on the base run under CompletedP
    config.optim.completedp = CompletedPOptimConfig(
        base_batch_size=BASE_BATCH, batch_size=BATCH,
        base_train_tokens=BASE_TOKENS, train_tokens=TOKENS,
    )
    # The production override "embeddings.weight" never matches under DDP (names carry a
    # "module." prefix); use a leading "*" if embeddings should get zero weight decay.
    config.optim.group_overrides = [
        OptimGroupOverride(params=["*embeddings.weight"], opts={"weight_decay": 0.0}),
        OptimGroupOverride(params=["*routed_experts.w_up_gate", "*routed_experts.w_down"], opts={}),
    ]
    return config
```

Before launching a script with CompletedP:
- `olmoe3_hero_mt.py` asserts that every `optim/LR (group i)` metric equals one cosine schedule. It
  fails with per-group multipliers and must be updated first.
- CompletedP only makes sense for runs that start from scratch; see Section 3.7.

### 3.5 Checking a run

- **Optimizer groups.** The `OLMoDDPOptimizerConfig` build log prints every group with its `lr`,
  `eps`, `weight_decay`, `betas`, the three `*_multiplier` keys and its parameter names. With the
  production overrides, the small hero gets 8 groups and the medium hero 14 (`alpha = 1`) or 15
  (`alpha = 0.5`); Section 1.6 lists them.
- **Learning rates.** The `optim/LR (group i)` metrics equal `lr_multiplier` times the schedule.
- **Residual multipliers.** Each `OLMoDDPTransformerBlock` has `attention_residual_alpha` and
  `feed_forward_residual_alpha` equal to `m_L^-alpha`. When folded,
  `block.attention_norm.output_scale` holds the value and `block._attention_add_alpha == 1.0`.
- **A quick offline check of the multipliers.** Build the model config on `meta` and call
  `olmo_core.optim.completedp.completedp_multipliers(model, CompletedPOptimConfig(...))`; that is
  how the table in Section 1.6 was made.

### 3.6 Per-module base hyperparameters (group overrides)

An override's `lr`, `eps`, `weight_decay` or `betas` replace the optimizer's base value for its
parameters, and CompletedP then scales them like any other base value. For example, a per-module LR
multiplier of 2 tuned at the base run for attention output projections:

```python
OptimGroupOverride(params=["*attention.w_out.weight"], opts={"lr": 2 * LR})
```

In the hero every sequence mixer is the block's `attention` module, so this pattern matches the
KDA output projections as well as the full-attention ones. Use the block indices to tell them
apart.

Override values are absolute, so compute them from the final base LR. Patterns must not overlap:
nothing deduplicates a parameter that matches two overrides
(`docs/PARAMETERIZATION_HANDOFF.md` Section 2.1).

### 3.7 Things to watch

- **Resumed runs.** The init only matters for runs that start from scratch. The residual
  multipliers and optimizer settings take effect immediately on resume. So switching CompletedP on
  or off, or changing `base_*` / `alpha`, for a checkpoint changes the model's forward function and
  every group's hyperparameters mid-training. Don't.
  - The DDP optimizer saves no group hyperparameters, so on resume everything comes from the config.
  - The generic `OptimConfig` path stores param groups in the checkpoint; the group count must
    match, and only `initial_lr` is re-applied from the config, so `eps`, `betas` and
    `weight_decay` come from the checkpoint.
- **More optimizer groups.** The real heroes get 8 (small) and 14-15 (medium) groups instead of
  2-3, so `optim/LR (group i)` indices change.
- **Betas.** Large `m_B / m_D` (big batch, short run) pushes the betas down and can make them
  negative (raises); small `m_B / m_D` pushes them toward 1.
- **`BatchSizeSchedulerCallback`** rescales only the LR (by `sqrt` of the batch ratio, which is the
  SDE rule for LR) and not the weight decay, epsilon or betas. It is inconsistent with CompletedP's
  other batch rules.
- **Not scaled:** global gradient clipping (`max_grad_norm`), the z-loss and load-balancing loss
  weights, and the LR schedule's shape.
- **Export.** The residual multipliers live in the model config and `LayerNorm.output_scale`, not in
  the weights. OLMo-core checkpoints are fine when loaded with the same model config, but HF
  conversion would drop the multipliers.

---

## 4. How correctness was tested

Sections 4.3-4.5 and 4.7 ran on one B300. The training runs of Section 4.6 ran on 8 B300s and its
static check on CPU. Software: torch 2.11,
FLA 0.4.1, flash-attn 2.8.2. `kernel_fun` is not installed, so KDA uses the FLA kernels
(`use_cute_kernel=False`). Standalone scripts need `PYTHONPATH=src`.

### 4.1 Unit tests (86 new)

`src/test/nn/transformer/completedp_test.py` (49):
- **Config.** Rejects bad `base_width`, `base_depth`, `alpha`; `residual_multiplier` for several
  depths and alphas.
- **Residual multipliers reach every block.**
  - Dense `default`, `reordered_norm`, `peri_norm` blocks including overrides; the user's config is
    not modified.
  - The DDP model, including a non-latent block-0 override.
  - A dense block computes `x + c * attn` then `h + c * mlp` (to fp32 tolerance).
  - A DDP block (peri-norm and pre-norm) gives `out_c - x == c * (out_1 - x)` for both branches.
    The multiplier is folded into the norms for peri-norm and applied as an add alpha for pre-norm.
- **Rejections.** A preset residual alpha, `init_method=nanochat`, tied embeddings, an unsupported
  block type, and the `normalized` and `moe` transformer types.
- **Roles.**
  - Dense model with an embedding norm.
  - Names keep a wrapper's `module.` prefix.
  - QK norms: full-width -> hidden vector, head-wise shared -> head-shared, per-head gains ->
    hidden vector.
  - DDP model: experts and latent projections hidden, the router a readout matrix, input norms.
  - KDA and GDN: `o_norm` head-shared; convs, `A_log`, `dt_bias` hidden vectors; KDA `f_proj_1`
    / `g_proj_1` readout matrices, `f_proj_2` / `g_proj_2` / `w_b` hidden (FLA-gated).
  - Readout multipliers (`alpha` 1 and 0.5): the router gets the same LR and weight decay as a
    hidden matrix with the same fan-in, and epsilon `m_L^-alpha` without the width factor.
- **Init.**
  - Every parameter of a CompletedP model is bit-identical to the `normal` init times its
    expected factor (`base_width` 128). This covers the dense model at `d_model` 128 and 256 and the
    DDP model at `d_model` 64 (CPU init).
  - Empirical stds match `0.02 * factor` within 6%; norm gains stay 1.
- **Precondition guard.** Under `normal`, every registry matrix has std `init_std` (within 10%):
  dense with elementwise gate and QK norm, DDP block with and without latent MoE,
  `FusedAttentionV2`, KDA, GDN. If a module ever stops initializing a matrix with `init_std`,
  CompletedP's rescaled init would be wrong and this fails.

`src/test/optim/completedp_test.py` (24):
- **Multipliers.** Every parameter's `lr`, `eps`, `weight_decay` multiplier matches Table 1, written
  out independently in the test. Model: 2x base width, 4x base depth, head-shared QK norms, an
  embedding norm. Cases: `alpha` 1 and 0.5, at base batch/duration, 4x batch (`s = 2`) and 4x
  tokens (`s = 0.5`).
- **Matches muP.** At the base depth, batch and duration, the learning rates equal muP's.
- **Built groups.** For `AdamWConfig` and `SkipStepAdamWConfig`: every parameter in exactly one
  group, values `base * multiplier`, `initial_lr == lr`, the `*_multiplier` keys, the scaled betas,
  one group per distinct triple, and a late `config.lr` change honored.
- **Composition with overrides.** An override's `lr` and `betas` are scaled; an embedding
  `weight_decay=0` override stays 0.
- **Betas.** Base, 4x batch -> (0.6, 0.8), 2x tokens -> (0.95, 0.975); 2x batch and 2x tokens ->
  unchanged; out-of-range raises.
- **Optimizer config.** Validation of the pairs and positivity.
- **Scheduler.** `CosWithWarmup.set_lr` keeps every group's ratio through warmup and decay.
- **Errors.** CompletedP optimizer on a non-CompletedP model; CompletedP model with a plain
  optimizer; `mup` and `completedp` together; `LionConfig`; `MuonConfig` on a CompletedP model.

Other files (13):
- `src/test/optim/moe_optimizer_test.py`: `OLMoDDPOptimizerConfig.build_groups` on a
  `module.`-prefixed wrapper (values, betas, an embedding override), and that a CompletedP model
  without `completedp` raises.
- `src/test/nn/ddp/block_test.py`: the DDP block stores the residual alphas (default and set).
- `src/test/nn/layer_norm_test.py`: `output_scale` for `LayerNorm` and `RMSNorm`, with and without
  bias, shared and per-head weights. The output and the gain's gradient are exactly scaled.
  `supports_output_scale` is false for an affine-free `RMSNorm`, `NemotronRMSNorm` and `L2Norm`.

Run:

```bash
pytest -v src/test/nn/transformer/completedp_test.py src/test/optim/completedp_test.py \
  src/test/optim/moe_optimizer_test.py src/test/nn/ddp/block_test.py src/test/nn/layer_norm_test.py
```

### 4.2 Regression suites

`src/test/optim`, `src/test/nn/transformer`, `src/test/nn/ddp`, `src/test/nn/layer_norm_test.py`,
`src/test/nn/attention` and `src/test/train/train_module`: 692 passed, 1083 skipped (multi-GPU and
kernel-specific), 40 xfailed, 3 failed. The 3 failures also fail on a clean `git archive HEAD` copy:
- `model_test.py::test_small_ngpt_builder_config` (2 cases);
- `headwise_qk_checkpoint_test.py::test_distributed_checkpoint_roundtrip`.

Static checks: isort, black and ruff are clean on every new and changed file. mypy reports only
the pre-existing error at `src/olmo_core/nn/moe/v2/routed_experts.py:940`.

### 4.3 Dense coordinate check

`src/examples/completedp/coord_check.py` trains a `llama_like` model and records the RMS change of
four activations after 10 AdamW steps on one fixed batch: the embedding output, the middle block,
the last block and the logits.
- Model: pre-norm by default, head dim 64 (heads grow with width), head-wise QK norm, fp32.
- Training: lr 3e-3, betas (0.9, 0.95), no weight decay, batch 8 x 128, vocab 1024.
- Sweeps: width 128-1024 at depth 4, and depth 2-32 at width 256. `base_width` 128,
  `base_depth` 2.
- "Exponent" is the slope of log(change) against log(width) or log(depth); 0 means
  width/depth-independent.

| Exponent of the RMS change | embedding | middle block | last block | logits |
|---|---|---|---|---|
| SP, width | -0.04 | +2.28 | +2.22 | +0.51 |
| SP, depth | +0.08 | +1.64 | +1.86 | +0.29 |
| CompletedP alpha=1, width | -0.01 | -0.03 | -0.02 | -0.21 |
| CompletedP alpha=1, depth | +0.07 | -0.23 | -0.10 | -0.07 |
| CompletedP alpha=0.5, width | -0.01 | -0.04 | -0.03 | -0.12 |
| CompletedP alpha=0.5, depth | +0.16 | +0.13 | +0.26 | +0.28 |

Raw values for two rows:
- SP middle block over width: 0.149, 1.50, 6.78, 17.5.
- CompletedP alpha=1 last block over depth: 0.048, 0.043, 0.039, 0.037, 0.036, approaching a limit.

**The logits' mild decline with width under CompletedP is a finite-width effect, not a bug.** The
width sweep was extended to 4096 and repeated with the LM head zero-initialized:
- default init: 0.470, 0.432, 0.359, 0.308, 0.319, 0.281 (exponent -0.15, flattening from 1024 up);
- zero init: exponent +0.02.

With the LM head zero-initialized, the logits' change is width-independent. With the paper's `1/d`
LM-head init, a term from the initial LM-head weights adds to it, and that term shrinks with width.

Three variants (flags `--block peri_norm --embedding-norm` and `--fresh-batches`) separate the
effect of peri-norm from the effect of training on fresh random tokens. "Fresh random tokens" means
a new uniformly random batch every step, so there is nothing to learn. The activations are still
measured on the fixed batch.

| Variant, exponent (hidden = last block) | SP width hidden / logits | SP depth hidden | CompletedP a=1 width hidden / logits | CompletedP a=1 depth hidden |
|---|---|---|---|---|
| pre-norm, fixed batch (above) | +2.22 / +0.51 | +1.86 | -0.02 / -0.21 | -0.10 |
| peri-norm + embedding norm, fixed batch | +0.03 / +0.54 | +0.93 | +0.03 / -0.16 | -0.11 |
| pre-norm, fresh random tokens | +1.97 / +0.57 | +1.30 | -0.45 / -0.23 | +0.24 |
| peri-norm + embedding norm, fresh random tokens | +0.05 / +0.54 | +0.96 | -0.14 / -0.40 | -0.12 |

- Peri-norm on its own does not break CompletedP.
- Fresh random tokens make CompletedP's changes shrink with width. With nothing to learn, the
  updates are noise and add up incoherently, which is not the regime muP-style rules are derived
  for. This explains the first hero coordinate check (Section 4.5).

### 4.4 Hero-architecture hard checks

`src/examples/completedp/hero_check.py` trains hero-like models with the production stack: the
real `Trainer`, `OLMoDDPTrainModule` and `OLMoDDPOptimizerConfig`.
- **Optimizer settings.** lr 8e-4, eps 1e-8, weight decay 0.1 (0 for embeddings through
  `*embeddings.weight`), betas (0.9, 0.95), the production routed-expert override, `compile=True`,
  `sigma_factor=6`, gradient clipping at 1.0.
- **Training.** fp32 gradient accumulation and reduction, z-loss 1e-5, `WSD(warmup=5, decay=5)`,
  20 steps, sequence length 4096, 2 sequences per step.
- **Models.** The hero block builders from `olmoe3_small_medium_models.py` with 64 experts (top-8),
  `d_model / 128` heads, block 0 dense, full attention every 8th block, peri-norm, latent MoE.
  QK-norm gains are shared (the `build_model_config` default); the unit tests cover per-head gains.
- **Sizes.** `d512-L8` (the base: `base_width` 512, `base_depth` 8, `alpha` 1), `d1024-L8`,
  `d2048-L8`, `d512-L16`, `d512-L32`.
- **Duration.** CompletedP runs train for twice the base duration (`m_D = 2`), so every LR and
  weight decay is x0.707, every epsilon x1.414 and the betas become (0.95, 0.975). This exercises
  the SDE rules.
- **Precision.** `compile_model=False`, so forward hooks can capture activations; KDA uses the FLA
  kernels.

Hard checks, all asserted inside a trainer callback:

1. Every block's residual multipliers equal `(n_layers / 8)^-alpha` (1 under SP). Every hidden
   matrix and the LM head start within 5% of `0.02 * factor` (0.02 under SP).
2. Every trainable parameter is in exactly one group. The group's `lr` / `eps` / `weight_decay`
   multipliers, `initial_lr`, `eps`, `weight_decay` and both betas equal the values computed in the
   script from the parameter's name and tensor layout. That calculation is independent of the
   library's role registry.
3. At every step, every group's LR equals its multiplier times the scheduled LR (relative 1e-6).
4. The first optimizer step. Before it, the callback captures each parameter's fp32 gradient and
   master weight. After it, using the clip coefficient `min(1, 1 / (grad_norm + 1e-6))`, it checks:
   - `exp_avg == (1 - beta1) * g` and `exp_avg_sq == (1 - beta2) * g^2` (relative 1e-5). This
     verifies both betas actually used.
   - `W_new == W_old * (1 - lr * wd) - lr * g / (|g| + eps)` elementwise, within
     `2^-18 * (|W_old| + lr)`. This verifies `lr`, `weight_decay` and `eps` actually used.

   The tolerance comes from computing the bias corrections `1 - beta^step` in fp32: with beta close
   to 1, rounding beta to fp32 moves `1 - beta` by up to about 1e-6 relative.
   - The library's own AdamW step, run eagerly on CPU with betas (0.95, 0.975), uses 0.97x of a
     `2^-20` tolerance.
   - The compiled GPU kernels use up to 2.2x of it, hence `2^-18`.
   - The check also counts how many elements the base epsilon (1e-8, i.e. without CompletedP's
     epsilon scaling) would put outside the tolerance. A non-zero count shows the check is
     sensitive to epsilon.

Results. Every run passed every check, in both rounds (fresh random tokens, then a fixed batch for
the coordinate check), plus 3 runs with `alpha = 0.5` and a rerun with the norm fold active: 24
runs. First round:

| Run | Groups | Tensors | worst exp_avg / exp_avg_sq error | worst weight error / tolerance | elements rejecting the base eps |
|---|---|---|---|---|---|
| d512-L8 CompletedP | 7 | 207 | 7.0e-08 / 1.1e-07 | 0.55 | 210,579,703 |
| d1024-L8 CompletedP | 10 | 207 | 7.0e-08 / 1.0e-07 | 0.55 | 767,160,702 |
| d2048-L8 CompletedP | 11 | 207 | 7.1e-08 / 1.0e-07 | 0.55 | 3,049,402,132 |
| d512-L16 CompletedP | 8 | 415 | 7.1e-08 / 1.1e-07 | 0.55 | 399,527,497 |
| d512-L32 CompletedP | 8 | 831 | 7.1e-08 / 1.1e-07 | 0.55 | 834,956,703 |
| each SP run | 3 | same | at most 7.1e-08 / 1.1e-07 | 0.18-0.19 | 0 (eps not scaled) |

The fixed-batch round gave the same figures (e.g. d1024-L8: 7.2e-08 / 9.8e-08, 0.55, 787,495,178),
and the `alpha = 0.5` runs too (d512-L16: 0.56).

**Rerun after the readout change.** The 8 CompletedP runs (5 sizes at `alpha = 1`, 3 depths at
`alpha = 0.5`, fixed batch) were rerun with routers, `f_proj_1` and `g_proj_1` as readout
matrices. All passed every check:

| Run | Groups | Tensors | worst exp_avg / exp_avg_sq error | worst weight error / tolerance | elements rejecting the base eps |
|---|---|---|---|---|---|
| d512-L8, alpha 1 | 7 | 207 | 7.0e-08 / 1.0e-07 | 0.55 | 224,134,723 |
| d1024-L8, alpha 1 | 10 | 207 | 7.0e-08 / 1.1e-07 | 0.55 | 791,251,749 |
| d2048-L8, alpha 1 | 11 | 207 | 7.1e-08 / 1.0e-07 | 0.55 | 3,156,133,124 |
| d512-L16, alpha 1 | 8 | 415 | 7.1e-08 / 1.1e-07 | 0.55 | 413,200,914 |
| d512-L32, alpha 1 | 8 | 831 | 6.9e-08 / 1.1e-07 | 0.55 | 859,643,937 |
| d512-L8, alpha 0.5 | 7 | 207 | 7.0e-08 / 1.0e-07 | 0.55 | 224,139,117 |
| d512-L16, alpha 0.5 | 8 | 415 | 7.1e-08 / 9.7e-08 | 0.56 | 49,848,034 |
| d512-L32, alpha 0.5 | 8 | 831 | 7.1e-08 / 1.1e-07 | 0.56 | 801,812,127 |

The group counts did not change. At the base depth the readout matrices have exactly the LM
head's multipliers (output-like in width, depth factors 1), so they share its group. In the depth
runs the width is the base (`m_N = r = 1`), so they have the multipliers of every other matrix
with fan-in `d_model` and share that group.

The per-role multiplier table printed by `report` matched Table 1 for every role at every size.
Examples:
- **2x width.** Projections from `d_model` get lr x0.354 (= 0.707 / 2), eps x0.707 (= 1.414 / 2) and
  wd x1.41 (= 0.707 * 2). Here 0.707 and 1.414 are the duration factors for `m_D = 2`.
- **Head-shared gains.** The shared QK-norm gains and KDA `o_norm` keep eps x1.41 at every width but
  halve it at 2x depth.
- **LM head.** Its eps stays x1.41 everywhere.
- **Readout matrices** (after the change). At 2x width the router, `f_proj_1` and `g_proj_1` get
  lr x0.354 and wd x1.41 like the other projections from `d_model`, but keep eps x1.41 like the LM
  head. At 2x depth their eps halves like every in-block parameter's.

### 4.5 Hero coordinate check

Same harness, recording the RMS change of the embedding output, block 1, the middle block, the
last block and the logits on the training batch after 20 steps. Width exponents use d512-d2048 at
depth 8; depth exponents use depth 8-32 at d512.

**First round, fresh random tokens.**
- SP: width logits +0.50; depth middle/last block +0.59/+0.60.
- CompletedP alpha=1: width logits -0.60, hidden -0.11 to -0.20; depth last block -0.58.

These are the noise-regime effects from Section 4.3, so the harness was switched to one fixed,
repeated batch.

**Fixed batch:**

| Site | d512-L8 | d512-L16 | d512-L32 | d1024-L8 | d2048-L8 | width exp | depth exp |
|---|---|---|---|---|---|---|---|
| SP last block | 5.28 | 8.12 | 12.35 | 5.72 | 5.65 | +0.05 | +0.61 |
| SP logits | 1.45 | 1.45 | 1.46 | 2.34 | 1.11 | -0.19 | +0.00 |
| CompletedP a=1 block 1 | 1.51 | 0.66 | 0.34 | 1.49 | 1.49 | -0.01 | -1.07 |
| CompletedP a=1 middle block | 3.37 | 2.32 | 1.71 | 3.37 | 3.37 | -0.00 | -0.49 |
| CompletedP a=1 last block | 5.12 | 3.91 | 3.10 | 5.15 | 5.14 | +0.00 | -0.36 |
| CompletedP a=1 logits | 0.90 | 0.85 | 0.87 | 0.80 | 0.76 | -0.12 | -0.02 |
| CompletedP a=0.5 middle block | 3.37 | 3.26 | 3.33 | | | | -0.01 |
| CompletedP a=0.5 last block | 5.12 | 5.25 | 5.39 | | | | +0.04 |
| CompletedP a=0.5 logits | 0.89 | 0.85 | 0.88 | | | | -0.01 |

The CompletedP rows are from the rerun after the readout change. The SP rows are from the earlier
round; SP is unaffected by the change. Before the change the CompletedP values were the same to
within 0.03 (the largest change: d2048-L8 last block 5.17, now 5.14), and every exponent within
0.01.

What this shows:
- **Width.** CompletedP is flat for every hidden activation on the hero. The logits show the same
  mild finite-width decline as the dense model.
- **Depth, alpha=1.** The residual stream's change shrinks as about `L^-0.36`, and the decline does
  not flatten between L16 and L32. Under SP the hero grows as `L^0.61`, compared with `L^0.93` for
  the dense peri-norm model. CompletedP alpha=1 removes about one power of L from SP. A plausible
  reading is that the hero's per-block updates add up only partly coherently, while alpha=1
  assumes they add up coherently. The mechanism was not established.
- **Depth, alpha=0.5.** The profile is flat.
- **Block 1** shrinks roughly as `L^-alpha` (exponents -1.07 for `alpha = 1`, -0.65 for
  `alpha = 0.5`). That is expected: it contains a single block's contribution.
- **The logits** are flat in depth under both alphas.

These activation changes do not by themselves show which alpha transfers the optimal LR better on
the hero; an LR sweep at two depths would (Section 5).

### 4.6 Production small and medium heroes

This section checks the exact production configs and records how CompletedP actually scaled every
parameter from the small hero (the base) to the medium hero. It compares that with the scaling
Table 1 predicts. There were two rounds:
- **8-GPU training runs, before the readout change.** At that point routers, `f_proj_1` and
  `g_proj_1` were still hidden matrices. Hard checks 1-4 of Section 4.4 ran on every rank,
  including the step-1 recomputation under expert parallelism.
- **A static check, after the readout change.** By then the host had a single GPU, and the medium
  hero (42.8B parameters; 95 GB per GPU with 8-way expert parallelism) does not fit on one. So the
  real configs were checked without training:
  - built on `meta` as the train module does, then the real `init_weights` run on CPU;
  - the optimizer groups built by the same `OLMoDDPOptimizerConfig` as the 8-GPU runs;
  - hero_check's `CompletedPCheck` callback then ran hard checks 1 and 2 on them: init stds,
    residual multipliers, and every tensor's group values. It was driven by a small script that
    is not checked in (Section 4.8).

  Hard checks 3 and 4 (the per-step LR and the step-1 recomputation) ran with the new
  classification only on the hero-like models (Section 4.4). The 8-GPU commands of Section 4.8
  would run them on the real configs.

Setup of the 8-GPU runs (`hero_check.py run small cp`, `run medium cp`, `run medium cp --alpha
0.5`, each with `torchrun --nproc-per-node=8` on 8 B300s):
- **Models.** `build_model_config("small")` and `build_model_config("medium")`: 512 routed experts
  (top-16), latent MoE, KDA plus gated full attention, peri-norm, and the per-head QK-norm gains of
  `olmoe3_small_integration.py`. KDA uses the FLA kernels (`kernel_fun` is not installed).
- **Parallelism.**
  - Small: 8-way DDP, with optimizer states sharded over the 8 ranks.
  - Medium: expert parallelism over the 8 ranks (64 experts per GPU) on the synchronized
    all-to-all path `sync_1d`. The production `rowwise_nvshmem` path needs a CUDA extension that
    cannot be built here (no `nvcc`).
- **CompletedP.** `CompletedPConfig(base_width=1024, base_depth=16, alpha=...)` on both models.
  `CompletedPOptimConfig` with the base batch size and duration equal to the run's, so `s = 1` and
  the betas stay `(0.9, 0.95)`. The batch/duration rules were exercised in Section 4.4.
- **Optimizer and training.** The production settings of Section 4.4: lr 8e-4, eps 1e-8, weight
  decay 0.1 (0 for embeddings through `*embeddings.weight`), the routed-expert override,
  `compile=True`, gradient clipping at 1.0. Then `WSD(warmup=5, decay=5)`, 20 steps, 2 sequences of
  4096 tokens per rank (65,536 tokens per step), random tokens, `compile_model=False`.
- **Checks.** Hard checks 1-4 of Section 4.4 run on every rank.
  - The step-1 recomputation uses each rank's shard of the optimizer state and of the gradient,
    narrowed the way the optimizer does it. Expert gradients are divided by the EP size, as the
    optimizer does.
  - Failures are gathered from all ranks.
  - The init stds of routed experts are measured on each rank's local experts.
  - There is no coordinate check for these runs.
- **Runtime.** About 9 minutes for the small hero and 10-11 minutes per medium run. The static
  check takes about 4 minutes (small) and 13 minutes (medium), almost all of it the CPU init.

**Hard checks, 8-GPU runs (before the readout change).** All three runs passed every check, at
every step:

| Run | Parallelism | Groups | Tensors | Residual multiplier, measured (expected) | Worst step-1 exp_avg / exp_avg_sq error | Worst step-1 weight error / tolerance | Elements rejecting the base eps |
|---|---|---|---|---|---|---|---|
| small (the base) | DDP x8 | 8 | 415 | 1 (1) on 32 branches | 7.2e-08 / 1.2e-07 | 0.18 | 0 (eps = base eps) |
| medium, alpha = 1 | EP x8 | 13 | 623 | 0.6667 (0.6667) on 48 branches | 7.1e-08 / 1.1e-07 | 0.19 | 42,244,644,160 |
| medium, alpha = 0.5 | EP x8 | 14 | 623 | 0.8165 (0.8165) on 48 branches | 7.1e-08 / 1.1e-07 | 0.20 | 42,183,601,097 |

Notes on the table:
- **Folded multiplier.** In every block the multiplier was folded into both post-branch RMSNorms:
  `output_scale` equals the multiplier and the residual add's alpha is 1.
- **Group LRs.** Every group's LR equaled its multiplier times the schedule at all 20 steps.
- **Epsilon sensitivity.** The last column sums over ranks, counting replicated tensors once. In
  the medium runs it shows that the step-1 check would have failed almost everywhere had the
  optimizer used the unscaled base epsilon.

**Hard checks 1-2, static check (after the readout change).** All three passed:

| Config | Groups | Tensors | Init stds checked | Residual multiplier, measured (expected) |
|---|---|---|---|---|
| small (the base) | 8 | 415 | 245 | 1 (1) on 32 branches |
| medium, alpha = 1 | 14 | 623 | 369 | 0.6667 (0.6667) on 48 branches |
| medium, alpha = 0.5 | 15 | 623 | 369 | 0.8165 (0.8165) on 48 branches |

The multiplier was again folded into the norms in every block. The medium hero has one group more
than before the change: the 65 readout tensors.

**Scaling from small to medium, measured vs Table 1** (static check, current classification).
Each cell is `measured / expected` (medium value over small value). Expected values use
`m_N = 1536/1024 = 1.5`, `m_L = 24/16 = 1.5` and each matrix's own `r = fan_in / 1024`. Because
the small hero is the base, a matrix's ratio uses `r_medium / r_small = fan_in_medium /
fan_in_small`.
- The lr, eps and wd ratios agree to better than 1e-9.
- The init-std ratios agree within 0.2%; they are measured on the initialized weights.
- The 8-GPU runs gave the same lr, eps and wd ratios for every row except the readout matrices,
  which they treated as hidden matrices, and init-std ratios within 0.3%.

| Parameters (fan-in small -> medium) | Table 1 rule: lr; eps; wd; init std | lr, alpha 1 | lr, alpha 0.5 | eps, alpha 1 | eps, alpha 0.5 | wd | init std |
|---|---|---|---|---|---|---|---|
| residual branches (48) | `m_L^-alpha` | 0.6667 / 0.6667 | 0.8165 / 0.8165 | | | | |
| hidden matrices with fan-in `d_model` (1024 -> 1536): KDA `w_q`, `w_k`, `w_v`, `w_b`; attention `w_q`, `w_k`, `w_v`, `w_g`; `latent_down_proj`; shared experts; routed `w_down` | `m_L^(a-1)/r`; `m_L^-a/m_N`; `r`; `r^-1/2` | 0.6667 / 0.6667 | 0.5443 / 0.5443 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 1.5 / 1.5 | 0.816 / 0.816 |
| readout matrices with fan-in `d_model` (1024 -> 1536): routers (512 outputs), KDA `f_proj_1`, `g_proj_1` (256 outputs) | `m_L^(a-1)/r`; `m_L^-a`; `r`; `1/r` | 0.6667 / 0.6667 | 0.5443 / 0.5443 | 0.6667 / 0.6667 | 0.8165 / 0.8165 | 1.5 / 1.5 | 0.668 / 0.667 |
| attention `w_out` (1024 -> 2048) | as hidden | 0.5 / 0.5 | 0.4082 / 0.4082 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 2 / 2 | 0.707 / 0.707 |
| KDA `w_out` (2048 -> 4096) | as hidden | 0.5 / 0.5 | 0.4082 / 0.4082 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 2 / 2 | 0.706 / 0.707 |
| KDA `f_proj_2`, `g_proj_2` (256 -> 256) | as hidden | 1 / 1 | 0.8165 / 0.8165 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 1 / 1 | 0.999 / 1 |
| `latent_up_proj`, routed `w_up_gate` (512 -> 768) | as hidden | 0.6667 / 0.6667 | 0.5443 / 0.5443 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 1.5 / 1.5 | 0.817 / 0.816 |
| block 0 dense `w_down` (8192 -> 12288) | as hidden | 0.6667 / 0.6667 | 0.5443 / 0.5443 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 1.5 / 1.5 | 0.817 / 0.816 |
| hidden vectors: norm gains, per-head QK gains, `A_log`, `dt_bias`, `ssmax_scale`, convs, biases | `m_L^(a-1)`; `m_L^-a/m_N`; 1; unchanged | 1 / 1 | 0.8165 / 0.8165 | 0.4444 / 0.4444 | 0.5443 / 0.5443 | 1 / 1 | - |
| KDA `o_norm` (head-shared) | `m_L^(a-1)`; `m_L^-a`; 1; unchanged | 1 / 1 | 0.8165 / 0.8165 | 0.6667 / 0.6667 | 0.8165 / 0.8165 | 1 / 1 | - |
| `embeddings`, `embedding_norm` | 1; `1/m_N`; 1; 1 | 1 / 1 | 1 / 1 | 0.6667 / 0.6667 | 0.6667 / 0.6667 | 1 / 1 | 1.000 / 1 |
| `lm_head.w_out` (1024 -> 1536) | `1/r`; 1; `r`; `1/r` | 0.6667 / 0.6667 | 0.6667 / 0.6667 | 1 / 1 | 1 / 1 | 1.5 / 1.5 | 0.666 / 0.667 |
| `lm_head.norm` | 1; 1; 1; unchanged | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | - |

The absolute multipliers on the base values (for example small KDA `w_out` at lr x0.5, wd x2, and
medium at lr x0.25, wd x4) equal the predicted values in Section 1.6 for every row, in all three
static checks.

**Measured init std vs expected** (static check). The expected value is
`0.02 * factor * 0.98658`; 0.98658 is the std of a unit normal truncated at +/-3. Each cell gives
the mean over the tensors of that kind. Over all initialized matrices the measured stds sit about
0.5% below the expected value on average (bf16 rounding), and no tensor is more than 1.7% off. The
8-GPU runs gave the same values to within 0.2% for every row except the readout matrices.

| Matrices (fan-in small -> medium) | small, measured (expected) | medium, measured (expected) |
|---|---|---|
| hidden, fan-in `d_model` (1024 -> 1536), 125 / 189 tensors | 0.01963 (0.01973) | 0.01603 (0.01611) |
| readout, fan-in `d_model` (1024 -> 1536), 43 / 65 tensors | 0.01963 (0.01973) | 0.01308 (0.01315) |
| attention `w_out` (1024 -> 2048) | 0.01963 (0.01973) | 0.01388 (0.01395) |
| KDA `w_out` (2048 -> 4096) | 0.01388 (0.01395) | 0.00981 (0.00987) |
| KDA `f_proj_2`, `g_proj_2` (256 -> 256) | 0.03926 (0.03946) | 0.03926 (0.03946) |
| `latent_up_proj`, routed `w_up_gate` (512 -> 768) | 0.02776 (0.02790) | 0.02267 (0.02278) |
| block 0 dense `w_down` (8192 -> 12288) | 0.00694 (0.00698) | 0.00567 (0.00570) |
| `lm_head.w_out` (1024 -> 1536) | 0.01963 (0.01973) | 0.01308 (0.01315) |
| `embeddings` (unchanged) | 0.01963 (0.01973) | 0.01963 (0.01973) |

The readout matrices now start at the LM head's std, 0.0131 on the medium hero, instead of the
0.0160 of the other hidden matrices with fan-in `d_model`.

**Optimizer groups as built** (static check; the 8-GPU runs additionally put the two routed-expert
groups in the expert-parallel partition).
- **Small: 8 groups.**
  - one group of 326 tensors with every multiplier 1, which includes the 43 readout tensors;
  - KDA `f_proj_2`/`g_proj_2` (lr x4, wd x0.25);
  - KDA `w_out` (lr x0.5, wd x2);
  - the dense `w_down` (lr x0.125, wd x8);
  - `latent_up_proj` (lr x2, wd x0.5);
  - the embeddings (wd 0);
  - the two routed-expert groups.
- **Medium: 14 groups (alpha = 1) or 15 (alpha = 0.5)**, as predicted in Section 1.6.
  - The 65 readout tensors form their own group: lr x0.6667, eps x0.6667, wd x1.5 (alpha = 1) or
    lr x0.5443, eps x0.8165, wd x1.5 (alpha = 0.5).
  - The other epsilon multipliers are 0.4444 for hidden tensors, 0.6667 for the embeddings,
    embedding norm and `o_norm`, and 1 for the LM head. With alpha = 0.5 they are 0.5443 for
    hidden tensors and 0.8165 for `o_norm`.

Run `report` to print the full group lists.

### 4.7 Throughput and MFU

The concern was that CompletedP slows pretraining. Levers:
- the init is one-time;
- the forward pass gains only the residual multiplier;
- the optimizer gets more groups with different constants;
- communication is unchanged.

**Trainer A/B.** `hero_check.py bench d1024-L16 {sp,cp}`: compiled model and optimizer, 60 steps of
4 x 4096 tokens, medians over steps 21+. Arms ran sequentially, alternating.

| Run | Groups | Tokens/s | MFU (%) | Optimizer step (ms) |
|---|---|---|---|---|
| SP #1 | 3 | 18,808 | 2.62 | 49.8 |
| CompletedP #1 | 11 | 17,980 | 2.51 | 50.6 |
| SP #2 | 3 | 18,712 | 2.61 | 51.7 |
| CompletedP #2 | 11 | 20,951 | 2.92 | 40.4 |

- The two CompletedP runs differ by 16% and per-step noise is about 10%, so this cannot resolve
  small effects. MoE routing on a small model makes step times noisy.
- No regression is visible.
- Eleven groups instead of three did not raise the optimizer step time.
- MFU is low because this model is small for a B300 at 16K tokens per step. That makes the test
  conservative: fixed per-step overheads weigh more than at production scale.
- This was measured before the readout change. The change adds one optimizer group at d1024-L16
  (12 instead of 11) and nothing to the forward pass, so it was not re-measured.

**Block microbenchmark (not checked in).** Recipe:
- **Build.** Take the d1024 KDA+MoE block from `hero_check.build_model_config("d1024-L16",
  completedp=False).block`, set both residual alphas, build it on CUDA and cast it to bf16.
  Initialize it with `InitMethod.normal` (`init_attention` and `init_moe_v2`).
- **Prepare.** Replace the routers with `_install_forced_router` from
  `src/test/nn/ddp/block_no_ep_test.py`, then call `apply_compile()`.
- **Time.** After 15 warmup iterations, time `block(x).float().square().mean().backward()` with
  CUDA events: 50 iterations per round, 5-7 rounds per alpha, alternating.
- **Profile.** Use `torch.profiler` (CUDA activity, 10 iterations), summing self device time per
  kernel name.

Results, before the fold (multiplier applied as `torch.add(..., alpha=c)`):

| Measure | alpha 1.0 | alpha 0.5 | difference |
|---|---|---|---|
| wall time, 4096 tokens | 12.56 ms | 12.66 ms | +0.79% |
| wall time, 32768 tokens | 19.57 ms | 19.63 ms | +0.31% |
| GPU kernel time, 4096 tokens | 2852 us | 2860 us | +7 us (+0.25%) |

Kernel counts were identical: the scale was already fused into existing Triton kernels. Only one
fused backward kernel got slower, by 5 us. At 4096 tokens the GPU is busy only about 23% of wall
time, so the benchmark is host-bound.

After folding the multiplier into the branch RMSNorm's gains, the GPU kernel time was 2851 us
(alpha 1) vs 2836 us (alpha 0.5). That is no increase; the remaining differences are GEMM-time
noise. A wall-clock rerun after the fold was too noisy to use: the host was loaded (load average
9-15), and rounds spread from 18.4 to 21.7 ms.

Conclusion: on the hero path (peri-norm, RMSNorm) the residual multiplier is free. Under the
`torch.add` fallback (pre-norm DDP blocks, other norm classes) it costs about 0.3% of a block's
forward and backward. The dense blocks' `ResidualStream` uses the same `torch.add` form and was not
measured separately. The optimizer groups cost nothing measurable. Multi-node runs were not
measured; communication is unchanged by design.

### 4.8 Reproducing

```bash
# Unit tests (about 15 s on CPU).
pytest -v src/test/nn/transformer/completedp_test.py src/test/optim/completedp_test.py

# Dense coordinate check (about 20 s per variant on one GPU).
PYTHONPATH=src python src/examples/completedp/coord_check.py
PYTHONPATH=src python src/examples/completedp/coord_check.py --alphas 1.0 --block peri_norm --embedding-norm
PYTHONPATH=src python src/examples/completedp/coord_check.py --alphas 1.0 --fresh-batches

# Hero checks on one GPU (6-8 minutes per run), then the report.
for spec in d512-L8 d1024-L8 d2048-L8 d512-L16 d512-L32; do
  for param in cp sp; do
    PYTHONPATH=src torchrun --nproc-per-node=1 src/examples/completedp/hero_check.py run $spec $param
  done
done
for spec in d512-L8 d512-L16 d512-L32; do
  PYTHONPATH=src torchrun --nproc-per-node=1 src/examples/completedp/hero_check.py run $spec cp --alpha 0.5
done
for param in sp cp sp cp; do   # about 10 minutes each
  PYTHONPATH=src torchrun --nproc-per-node=1 src/examples/completedp/hero_check.py bench d1024-L16 $param
done
PYTHONPATH=src python src/examples/completedp/hero_check.py report

# Production small and medium heroes on 8 GPUs (9-11 minutes each), then the scaling report.
OUT=/path/to/results
PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/completedp/hero_check.py run small cp --out-dir $OUT
PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/completedp/hero_check.py run medium cp --out-dir $OUT
PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/completedp/hero_check.py run medium cp --alpha 0.5 --out-dir $OUT
PYTHONPATH=src python src/examples/completedp/hero_check.py report --out-dir $OUT
```

`run` exits non-zero if a hard check fails, and `report` exits non-zero if any stored run failed.
Results go to `--out-dir` (default `/tmp/completedp_hero_check/`, one JSON per run).

**Static check of the real heroes (not checked in).** Used in Section 4.6 when only one GPU was
available. The script builds the hero config with `hero_check.build_hero_config` on `meta` and
runs `init_weights(device=cpu)`. It builds the groups with the `OLMoDDPOptimizerConfig` of
`hero_check._train` and fills in the defaults `torch.optim.Optimizer` would add. Then it calls
`CompletedPCheck.pre_train()` with a stand-in trainer, and writes the same JSON as `run`, so
`report` prints the scaling tables. In that report the step-1 columns read 0 and the rank count 0,
because no step was taken. The medium hero's weights take 86 GB of RAM in bf16.

```bash
OUT=/path/to/results
for args in "small 1.0" "medium 1.0" "medium 0.5"; do
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src:src/examples/completedp:src/examples/olmo_ddp \
    python static_hero_check.py $args $OUT
done
PYTHONPATH=src python src/examples/completedp/hero_check.py report --out-dir $OUT
```

**Where the results are.** All outside the repository, under
`/weka/oe-training-default/tanushy/completedp_hero_results/`:
- the 8-GPU runs of Section 4.6 at the top level, with logs in `logs/`;
- the static check of Section 4.6 in `v2_static/`, with `static_hero_check.py` next to it;
- the hero-like reruns after the readout change (Sections 4.4-4.5) in `hero_like_v2/`.

The earlier single-GPU results of Sections 4.4-4.5 and 4.7 were written to `/tmp` and lost; their
numbers are copied from the report output.

### 4.9 What these tests do not establish

- **Hyperparameter transfer.** No LR (or other hyperparameter) sweep was run. The checks verify the
  mechanics and activation scaling, not that the tuned optimum moves as intended. That needs a
  short LR sweep on real data at two or three sizes, which would also settle the choice of alpha
  for the hero.
- **The batch-size and duration rules** were verified mechanically (every group's values, applied
  in the step), not empirically.
- **Parallelism and precision.** Exercised before the readout change: 8-way DDP (small hero) and
  8-way expert parallelism on the synchronized `sync_1d` path (medium hero), Section 4.6. The
  change does not touch sharded tensors (routers and KDA projections are replicated), but the
  8-GPU runs have not been repeated with it. Not exercised: the production
  `rowwise_nvshmem` EP path, pipeline parallelism (the init rescale runs per model part), tensor
  parallelism, FP8 weight stores, multi-node runs, and throughput at more than one GPU.
- **Checkpoint resume** with CompletedP groups.
- **The exact small and medium heroes** were trained for 20 steps on random tokens (Section 4.6),
  before the readout change. That checks the mechanics, not training quality. With the change
  they were checked without training, so the step-1 recomputation has not been run on the real
  configs with the readout rules.
- **Independence of the expected values.** The expected values in the hero check were written by
  the same author as the implementation, from the same reading of Table 1. The role and fan-in
  classification is independent of the library's registry; the formulas are not.

---

## 5. Gotchas and possible follow-ups

1. **Choose alpha with an LR sweep.** Run the base LR grid at two depths (e.g. hero-like d1024 at
   16 and 32 layers) with `alpha = 1` and `alpha = 0.5`; the alpha whose optimal LR stays put wins.
   The coordinate check favors 0.5 on the hero.
2. **Base values must be fixed across compared runs** and tuned under CompletedP (Section 3.1).
3. **New modules or parameters.**
   - A new sequence mixer or block type raises until added to the allowlists in
     `validate_completedp_model` / `completedp_block_configs`.
   - Before adding a mixer, check that `normal` initializes all its matrices with `init_std`
     (extend `test_normal_init_precondition_*`), because the CompletedP init rescales it.
   - New matrices need a fan-in rule in muP's `_module_param_shapes`.
   - New gains shared across heads need adding to `_head_shared_param_names`.
   - New in-block matrices whose output size does not grow with width (a new router type, a
     fixed-rank bottleneck) need adding to `_readout_param_names`. So do the conditional cases in
     a config that fixes a size the hero lets grow: a fixed latent rank (`latent_down_proj`), fixed
     expert width (expert gate/up), fixed KV-head count (K/V), fixed head count (`w_b`). Otherwise
     they get the hidden init and epsilon.
4. **`output_scale` is only honored by `LayerNorm` and `RMSNorm`.** Other norm classes override
   `forward`, report `supports_output_scale() == False`, and the DDP block falls back to
   `torch.add`. Never set `output_scale` on them directly.
5. **Unsupported, and raising:** MoE v1 and hybrid blocks, nGPT, Nemotron, Mamba2, tied embeddings,
   non-`normal` init, Muon, Dion, Lion, Adam (non-decoupled weight decay), and combining with `mup`.
6. **Weight decay is coupled to LR.** CompletedP's weight-decay rules keep `lr * wd` fixed in width
   (each matrix's `r` cancels). They do not fix it in depth for `alpha < 1`, nor in batch/duration,
   where `lr * wd ~ m_B / m_D` by design (the AdamW timescale rule).
7. **Unembedding-norm epsilon** follows Table 1 (x1). If coordinate checks on the LM-head norm ever
   look off, a `1/m_N` rule would be the first thing to try.
8. **The hero LR audit** in `olmoe3_hero_mt.py` and the unmatched `embeddings.weight` override
   (Section 3.4) need fixing before CompletedP is used in hero scripts.
9. **Possible follow-ups:**
   - read the batch size and duration from the trainer and check them at `pre_train`;
   - make `BatchSizeSchedulerCallback` apply all of CompletedP's batch rules;
   - fold the multiplier in the dense `ResidualStream` blocks too;
   - teach the HF converters about the residual multipliers;
   - per-module init scales and per-depth multipliers (the paper's per-module search);
   - check in the block microbenchmark if throughput needs re-measuring.
