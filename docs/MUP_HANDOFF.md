# Handoff: muP (nanochat init + per-layer Adam learning rates)

Written 2026-10-18 against commit `9059a225c` (branch `tanushy/hero-12p4t-mt20-20261012`) plus
uncommitted changes. **Nothing has been committed.** Line numbers refer to the working tree at
the time of writing; re-grep before editing.

Companion docs:
- `docs/PARAMETERIZATION_HANDOFF.md`: how init and optimizer groups work in this codebase.
- `docs/NANOCHAT_INIT_HANDOFF.md`: the `InitMethod.nanochat` init, which is the muP init used here.

Notation: `lr` = the optimizer's base learning rate (or a group override's `lr`), `m` = a
parameter's muP learning-rate multiplier, `fan_in` / `fan_out` = a weight matrix's logical input /
output size.

---

## 0. Summary

muP has two halves:

| Half | Rule | Where it lives |
|---|---|---|
| Init | `std = (1/sqrt(fan_in)) * min(1, sqrt(fan_out/fan_in))` | `InitMethod.nanochat` (implemented earlier; unchanged by this work) |
| Learning rate (Adam) | `lr_l = lr * base_width / fan_in` for weight matrices | **new**: `MuPConfig`, set as `mup=` on an optimizer config |

Embeddings get `lr * embedding_lr_multiplier` (default 1.0). Element-wise parameters (norm gains,
biases, per-head scalars, depthwise conv taps) get `lr`.

Usage:

```python
from olmo_core.nn.transformer import InitMethod
from olmo_core.optim import AdamWConfig, MuPConfig, OLMoDDPOptimizerConfig

model_config.init_method = InitMethod.nanochat          # muP init (model config)
optim = AdamWConfig(lr=1.1e-3, mup=MuPConfig(base_width=1024))   # generic path
optim = OLMoDDPOptimizerConfig(lr=1.1e-3, ..., mup=MuPConfig(base_width=1024))  # hero path
```

The two halves are configured independently: the init on the model config, the LRs on the
optimizer config. The optimizer logs a one-time warning if muP LRs are enabled on a model whose
`init_method` is not `nanochat`.

**Update (2026-10-25): base-model mode.** `MuPConfig` now also accepts a base model instead of a
single `base_width`; see [Section 0.1](#01-base-model-mode-base_fan_ins). The rest of this document
describes the original `base_width` mode, which is unchanged.

### 0.1 Base-model mode (`base_fan_ins`)

With a fixed `base_width`, the base model is not trained at `lr` everywhere: any matrix whose
fan-in differs from `base_width` gets another LR even at the base size (e.g. an FFN down-projection
with fan-in `8 * d_model` gets `lr / 8`). In effect, each kind of matrix gets an arbitrary
base LR. Base-model mode fixes the base LR of every matrix at `lr` and only scales by how much each
matrix's fan-in changed relative to the base model:

```text
lr_l = lr * base_fan_in(role) / fan_in      (matrices)
```

where `base_fan_in(role)` is the fan-in of the same matrix in the base model. Embeddings and
element-wise parameters are unchanged. A group override's `lr` is still multiplied, so a group can
be given its own base LR. The fixed-`base_width` mode is the special case where every role's base
fan-in is `base_width`.

```python
from olmo_core.optim import MuPConfig

base_model = base_model_config.build(init_device="meta")  # the base size, e.g. a 275M rung
mup = MuPConfig.from_base_model(base_model)               # sets base_fan_ins; exactly one of
                                                          # base_width / base_fan_ins is allowed
```

`base_fan_ins` is a plain `{role: fan_in}` table stored in the config, so the optimizer never
builds the base model, and the table is visible in the saved config.

**Roles** (`named_matrix_roles` in `src/olmo_core/nn/transformer/mup.py`) name the same matrix in
models of different width and depth:

- Outside the blocks, the role is the parameter name without wrapper prefixes, e.g.
  `lm_head.w_out.weight`.
- Inside a block, the role is `blocks[<kind>].<path within block>`, e.g.
  `blocks[attention:KimiDeltaAttention+feed_forward].attention.w_out.weight`. The block kind lists
  the block's non-norm child modules in registration order, with the sequence mixer's class
  (the `attention` child). So linear-attention vs. full-attention blocks, and dense vs. MoE
  blocks, get different roles, while models with more layers share them.
- Roles ignore the `module.` wrapper prefix, activation-checkpointing (`_checkpoint_wrapped_module`)
  and `torch.compile` (`_orig_mod`) name segments, and FSDP's `FSDP<Class>` class swaps.

**Strictness:** `matrix_fan_ins_by_role` raises if two matrices with the same role have different
fan-ins in the base model (the role would be ambiguous), and `mup_lr_multipliers` raises if a
matrix's role is not in `base_fan_ins` (e.g. a block kind the base model doesn't have).

Tests: `test_named_matrix_roles_*` and `test_matrix_fan_ins_by_role_rejects_ambiguous_roles` in
`src/test/nn/transformer/mup_test.py`; `test_mup_config_rejects_invalid_bases` and
`test_mup_base_model_*` in `src/test/optim/mup_test.py`;
`test_build_groups_applies_mup_base_model_with_wrapper_prefix` in
`src/test/optim/moe_optimizer_test.py`.

---

## 1. Design decisions

### 1.1 Settled with the user

1. **Adam rule, not the SGD rule.** The user's formula, `lr_l ~ fan_out/fan_in`, is the
   spectral-condition rule for SGD. Its derivation: under SGD a matrix's update is
   `lr * gradient`, the gradient is rank-1 with spectral norm about `sqrt(fan_in/fan_out)`, and
   the spectral condition asks for an update of size about `sqrt(fan_out/fan_in)`. Adam
   normalizes each gradient entry to about +/-1, so its update has spectral norm about
   `lr * sqrt(fan_in * fan_out)`, which gives `lr ~ 1/fan_in`. This matches the Adam column of
   Table 3 in Tensor Programs V (input layer: constant; hidden and output layers: `1/fan_in`).
   The SGD rule under AdamW would leave square hidden matrices at the same LR at every width,
   i.e. standard parameterization. The codebase trains with AdamW, so the Adam rule was chosen.
2. **Constants fixed by a `base_width`.** muP only fixes how the LR scales with width.
   `base_width` anchors the absolute values: a matrix whose fan-in equals `base_width` gets
   exactly `lr`. The rejected alternative was the `mup` library's base-model approach (every LR
   equals `lr` exactly at the base model), which requires matching parameters to a base model by
   name and therefore the same depth and block layout. Small and medium hero differ in depth.
   - Standard practice uses a single model-wide multiplier `d_model / base_width`. This
     implementation uses **each parameter's own fan-in**. That is correct when dimensions don't
     scale together (hero small -> medium: `d_model` grows 1.5x, attention's query width 2x,
     KDA's 256-dim low-rank projections not at all), but it means that even at the base width
     not every LR equals `lr` (e.g. KDA `f_proj_2` gets 4x). The base LR must be re-tuned.

### 1.2 Defaults chosen in the plan (approved with it)

- Reuse `InitMethod.nanochat` as the muP init (identical formula).
- Multiplier 1 for embeddings (configurable through `embedding_lr_multiplier`), vectors and
  depthwise convs.
- Weight decay coupled by default: the optimizer applies `p *= 1 - lr_l * wd`, so decay scales
  with the multiplier (the `mup` library's `decoupled_wd=True`; its default,
  `decoupled_wd=False`, scales weight decay instead). `scale_weight_decay=True` sets
  `wd_l = wd / m`, which keeps `lr_l * wd_l` at its base value for every parameter (section 5.1).
- Strict: an unrecognized parameter with ndim >= 2 raises.
- Muon and Dion (`MatrixAwareOptimConfig`) raise if `mup` is set.
- No hero script changes.

---

## 2. High-level workflow

```text
model config (init_method=nanochat) --build/init_weights--> model
                                                              |
optimizer config (mup=MuPConfig(base_width))                  |
  build()                                                     |
    build_groups(model)  <------------------------------------+
      1. expand group_overrides globs; add the default group (existing code)
      2. mup_lr_multipliers(model):
           named_param_shapes(model) -> {name: ParamShape(kind, fan_in, fan_out)}
           kind -> m:  matrix: base_width / fan_in
                       embedding: embedding_lr_multiplier
                       vector: 1.0
      3. split_groups_by_lr_multiplier(groups, multipliers):
           each group -> one sub-group per distinct m
           sub-group opts = group opts + {"lr": group_lr * m, "lr_multiplier": m}
                            (+ "weight_decay": group_wd / m if scale_weight_decay)
    construct optimizer from the groups
    set group "lr" (tensor if compile=True) and "initial_lr" = that lr   (existing code)
each step:
  Scheduler.set_lr(group, trainer): lr = schedule(group["initial_lr"], step)  (existing code)
  optimizer step uses each group's own lr / weight_decay                       (existing code)
```

Key properties:

- **Per-group LRs already existed**; only multipliers and group generation are new. The existing
  pieces that make it work, all **unchanged**:
  - `build()` copies a group's `lr` into `lr` and `initial_lr`
    (`optim/config.py:220` generic; `optim/moe_optimizer.py:426` DDP).
  - `Scheduler.set_lr` scales from each group's own `initial_lr` (`optim/scheduler.py:79`), so
    warmup and decay keep the ratios.
  - `BatchSizeSchedulerCallback` multiplies every group's `initial_lr` by the same factor.
  - The DDP optimizer steps each group with its own `lr` (`moe_optimizer.py:2119`, inside
    `_step_foreach` at line 2077).
- **LRs are computed from the final `lr`** because splitting happens at `build()` time. Hero
  scripts set `config.optim.lr = LR` after building the optimizer config
  (e.g. `olmoe3_small_integration.py:229`); that late value is what gets multiplied.
- **Group overrides compose**: an override's own `lr` is multiplied, and its other options
  (`weight_decay`, `betas`, ...) are kept.
- **Call sites** (unchanged): `TransformerTrainModule` builds with
  `optim.build(self.model, strict=True)` (`train/train_module/transformer/train_module.py:239`);
  `OLMoDDPTrainModule` builds on the `MultiGroupDistributedDataParallel`-wrapped model
  (`ddp_train_module.py:342`), so parameter names carry a `module.` prefix. The registry walks
  whatever module it is given, so its names carry the same prefix.

---

## 3. Files and what each houses

### 3.1 `src/olmo_core/nn/transformer/mup.py` (new, 131 lines): the parameter-shape registry

Model-structure knowledge only; no optimizer logic.

| Line | Name | Purpose |
|---|---|---|
| 15 | `ParamKind` (`StrEnum`) | `matrix`, `embedding`, `vector`. |
| 39 | `ParamShape` (frozen dataclass) | `kind`, plus `fan_in` / `fan_out` for matrices (and embeddings: vocab -> d). |
| 49 | `_module_param_shapes(module)` | Shapes of the parameters a module owns **directly**, keyed by local name. Dispatches on module type (table below). |
| 95 | `named_param_shapes(model)` | Walks `model.named_modules()` and classifies every parameter. Unclaimed parameters with ndim <= 1 become `vector`; unclaimed ones with ndim >= 2 raise `NotImplementedError("muP has no rule for parameter ...")`. Shared parameters (tied embeddings) are classified once, by their first owner, like `named_parameters()`. The returned names and order match `model.named_parameters()`, including any wrapper prefix. |

Classification rules (`_module_param_shapes`, lines 58-92):

| Module type | Parameter(s) | Kind | fan_in -> fan_out |
|---|---|---|---|
| `nn.Embedding` | `weight` | embedding | `num_embeddings -> embedding_dim` |
| `nn.Linear` (incl. `MXFP8Linear`, attention/KDA/GDN/FFN/latent projections, LM head) | `weight` | matrix | `in_features -> out_features` |
| `nn.Conv1d`, depthwise (`groups == in_channels == out_channels`; e.g. `CausalConv1d`, Mamba2 conv) | `weight` | vector | - |
| `nn.Conv1d`, not depthwise | `weight` | raises (not claimed, ndim 3) | - |
| `LayerNorm` and subclasses | all (incl. 2-D per-head QK-norm gains) | vector | - |
| `RoutedExperts`, `SharedExperts` | `w_up_gate` / `w_down` | matrix | `d_model -> hidden_size` / `hidden_size -> d_model` |
| `RoutedExperts`, `SharedExperts` | `b_up_gate`, `b_down` (2-D) | vector | - |
| `MoERouterV2`, `MoERouter` (v1, e.g. `MoELinearRouter`) | `weight` (flattened 1-D) | matrix | `d_model -> num_experts` |
| `MoEMLPBase` (MoE v1 experts) | `w1`, `w3` / `w2` | matrix | `d_model -> hidden_size` / `hidden_size -> d_model` |
| anything else | ndim <= 1 (biases, `A_log`, `dt_bias`, `D`, `ssmax_scale`, sinks, FLA norm gains) | vector | - |

Notes:
- `fan_out` for fused `w_up_gate` is `hidden_size` (each half), mirroring the nanochat init. The
  Adam LR rule only uses `fan_in`, so fused fan-outs do not affect LRs.
- Under latent MoE, `routed_experts.d_model` is the latent width (512 on the small hero), while
  the router's `d_model` is the model width (1024).
- NemotronMoE and Mamba2 are covered by these rules (NemotronMoE reuses `MoERouterV2`,
  `RoutedExperts`, `SharedExperts`), even though `InitMethod.nanochat` raises for them.
- FP8 weight stores (`named_fp8_weight_stores`) are added to the DDP optimizer under the same
  names as their anchor parameters (e.g. `blocks.1.routed_experts.w_up_gate`,
  `blocks.7.attention.w_qkv.weight`), so they are found by name. Not exercised with FP8 enabled.

### 3.2 `src/olmo_core/optim/mup.py` (new, 150 lines): config, multipliers, group splitting

Optimizer-side logic; imports the registry. Imports `OptimGroupOverride` only under
`TYPE_CHECKING` (it builds new groups with `dataclasses.replace`), which avoids a circular import
with `optim/config.py`.

| Line | Name | Purpose |
|---|---|---|
| 25 | `LR_MULTIPLIER_FIELD = "lr_multiplier"` | Informational key recorded in every split group. |
| 29 | `MuPConfig(Config)` | Fields: `base_width: int` (line 45, required), `embedding_lr_multiplier: float = 1.0` (line 50), `scale_weight_decay: bool = False` (line 55). `__post_init__` (line 62) rejects `base_width <= 0`. |
| 67 | `mup_lr_multipliers(model, config)` | Logs a one-time warning (via `log_once`) if the first module with an `InitMethod`-typed `init_method` attribute is not `nanochat`. Returns `{name: m}` from `named_param_shapes`: matrix `base_width / fan_in`, embedding `embedding_lr_multiplier`, vector `1.0`. |
| 104 | `split_groups_by_lr_multiplier(groups, multipliers, config, *, lr, weight_decay=None)` | For each `OptimGroupOverride` (names already expanded), groups its parameters by multiplier in order of first appearance and emits one sub-group per multiplier with `lr = opts.get("lr", lr) * m` and `lr_multiplier = m`; with `scale_weight_decay`, `weight_decay = opts.get("weight_decay", weight_decay) / m`. Empty groups produce nothing. Raises `OLMoConfigurationError` if a parameter has no multiplier, or if `scale_weight_decay` is set and no weight decay is known. |

### 3.3 `src/olmo_core/optim/config.py` (modified): generic optimizer configs

Covers every `OptimConfig` subclass: `AdamWConfig`, `SkipStepAdamWConfig`, `AdamConfig`,
`LionConfig`, `SkipStepLionConfig`, and others.

| Line | Change |
|---|---|
| 29 | `from .mup import MuPConfig, mup_lr_multipliers, split_groups_by_lr_multiplier` |
| 85 | New field `OptimConfig.mup: Optional[MuPConfig] = None`. |
| 95-108 | New `_apply_mup(group_overrides, model)`: no-op if `mup` is unset; raises if the config has no `lr` attribute; otherwise calls the splitter with `lr=self.lr` and `weight_decay=getattr(self, "weight_decay", None)`. |
| 171 | In `build_groups`, after the default group is appended (line 170) and before groups become param dicts: `group_overrides = self._apply_mup(group_overrides, model)`. Group order: override groups first, then the default group, each expanded into its multiplier sub-groups. |
| 199 | `build()` pops `"mup"` from the kwargs so it isn't passed to the optimizer constructor. |
| 364 | `MatrixAwareOptimConfig.build()` raises `OLMoConfigurationError` if `mup` is set. |
| 370 | ...and pops `"mup"` from its kwargs. |

### 3.4 `src/olmo_core/optim/moe_optimizer.py` (modified): hero DDP optimizer config

`OLMoDDPOptimizerConfig` (line 119) is **not** an `OptimConfig` subclass, so it needs its own
wiring.

| Line | Change |
|---|---|
| 48 | `from .mup import MuPConfig, mup_lr_multipliers, split_groups_by_lr_multiplier` |
| 152 | New field `mup: Optional[MuPConfig] = None`. |
| 286-297 | In `build_groups`, after the default group is inserted first (line 284): multipliers are merged across all `model_parts`, then the groups are split with `lr=self.lr`, `weight_decay=self.weight_decay`. Group order: default group's sub-groups first, then each override's. |
| 362 | `build()` pops `"mup"`. |

`build()` calls `build_groups` twice, once for the dense (`pg="dp"`) partition and once for the
expert-parallel (`pg="ep_dp"`) partition, so both are split. The existing build log (lines
432-449) prints every group field except the parameters, so `lr` and `lr_multiplier` appear in
it automatically.

Things verified in this class that make many groups cheap:
- Optimizer state is keyed per parameter name.
- Flat sync buffers are keyed by partition (`_init_flat_model_param_buffers`, line 820), not by
  group. That function also rejects a parameter that appears in two groups (bf16 path); the
  split keeps groups disjoint.
- With `compile=True`, each group's `lr` is a tensor, so different values do not recompile the
  compiled `_step_foreach` (line 453).
- Groups returned by `build_groups` carry only their override keys; `OLMoDDPOptimizer` fills in
  defaults (`weight_decay`, `betas`, `eps`) itself. So a non-overridden group has no
  `weight_decay` key unless `scale_weight_decay` sets one.

### 3.5 `src/olmo_core/optim/__init__.py` (modified)

Exports `MuPConfig` (import at line 14, `__all__` at line 47). `mup_lr_multipliers` and
`split_groups_by_lr_multiplier` are imported from `olmo_core.optim.mup`.

### 3.6 Tests

- **`src/test/nn/transformer/mup_test.py`** (new): the registry.
  - Builders: `_dense_model` (line 43; `llama_like` with grouped-query attention),
    `_moe_model` (line 55; MoE v1 with `shared_mlp`, optional latent),
    `_ddp_block` (line 77; MoE-v2 block with latent option, shared expert, peri-norm and per-head
    QK-norm gains).
  - `test_named_param_shapes_dense` (116; headwise and elementwise gates),
    `_tied_embeddings` (137), `_keeps_wrapper_prefix` (145), `_moe` (158), `_ddp_block` (178),
    `_fused_attention_v2` (195), `_linear_attention` (212; KDA and GDN, skipped without FLA),
    `_raises_for_unknown_matrices` (228; raw 2-D parameter, non-depthwise conv).
  - **Init consistency guard**: `_assert_matrices_match_nanochat_init` (233) checks that every
    matrix's nanochat-initialized std equals `nanochat_std(fan_in, fan_out)` from the registry.
    Used by `test_param_shapes_match_nanochat_init_dense` (245), `_moe` (252), `_ddp_block`
    (259). If someone changes the init or the registry for a module and not the other, these
    fail.
- **`src/test/optim/mup_test.py`** (new): optimizer behavior on a small `llama_like` model
  (`_model`, line 29; d=128, FFN hidden 512, `BASE_WIDTH = 64`).
  - `test_mup_lr_multipliers` (45), `test_mup_config_rejects_nonpositive_base_width` (57).
  - `test_mup_param_groups` (63; `AdamWConfig` and `SkipStepAdamWConfig`): every parameter in
    one group, `lr = lr * m`, `initial_lr == lr`, one group per distinct multiplier, and a
    late `config.lr` change is honored.
  - `test_mup_composes_with_group_overrides` (80), `test_mup_weight_decay` (104; coupled and
    scaled), `test_mup_scheduler_keeps_ratios` (122; `CosWithWarmup.set_lr` through warmup and
    decay), `test_mup_split_requires_every_param` (135),
    `test_mup_rejected_for_matrix_aware_optimizers` (141; Muon),
    `test_mup_warns_without_nanochat_init` (146; clears the `log_once` cache first).
  - `scale_weight_decay` with base-model fan-ins and an embeddings `weight_decay=0.0`
    override (as in the scaling-ladders mainline ladder), each run for `AdamWConfig` and
    `SkipStepAdamWConfig` with and without foreach:
    `test_mup_base_model_scale_weight_decay` (stored `weight_decay = wd / m`, `lr * wd` fixed),
    `test_mup_weight_decay_applied` (a zero-gradient step, where Adam's update is exactly zero,
    shrinks each parameter by `lr * wd` scaled or `lr * m * wd` coupled),
    `test_mup_scheduler_keeps_decay` and `test_mup_scale_weight_decay_state_dict_roundtrip`
    (a loaded state dict's group `weight_decay` replaces the config's).
- **`src/test/optim/moe_optimizer_test.py`** (modified):
  `test_build_groups_applies_mup_with_wrapper_prefix` (line 28): `OLMoDDPOptimizerConfig` on a
  wrapper exposing `module.`-prefixed names, with a `*bias*` weight-decay override.

Run: `pytest -v src/test/nn/transformer/mup_test.py src/test/optim/mup_test.py src/test/optim/moe_optimizer_test.py`

### 3.7 `src/examples/mup/coord_check.py` (new) + empty `__init__.py`

The standard muP sanity test.
- `activation_changes(width, *, mup, args)` (line 35) trains one `llama_like` model
  (pre-norm, `head_dim = HEAD_DIM = 64`, so heads grow with width, torch attention backend, fp32)
  for `--steps` AdamW steps on one fixed random batch.
- It returns the per-coordinate RMS change of the embedding output (forward hook), the middle
  block's output (forward hook), and the logits.
- Standard parameterization uses `InitMethod.normal` and one LR; muP uses
  `InitMethod.nanochat` + `MuPConfig(base_width=--base-width)`.
- `width_exponent` (line 98) is the least-squares slope of log(change) vs log(width).
- `main` (line 110) prints one table per parameterization.

Run: `python src/examples/mup/coord_check.py` (one GPU, about 15 s). Defaults: widths
128/256/512/1024, `--base-width 128`, `--lr 3e-3`, 10 steps, 4 layers, vocab 1024, batch 8 x 128.

### 3.8 `CHANGELOG.md`

One `### Added` entry under `## Unreleased`.

### 3.9 Not changed

`nn/transformer/init.py` (the nanochat init), the schedulers, the train modules, and every hero
script.

---

## 4. Hero multipliers (`base_width=1024`)

Small hero (d=1024). This is the actual grouping produced by
`OLMoDDPOptimizerConfig.build_groups` with the hero's overrides:

| Group | m | lr at 1.1e-3 | Parameters |
|---|---|---|---|
| 0 | 1 | 1.1e-3 | everything with fan-in 1024 (KDA / attention `w_q`, `w_k`, `w_v`, `w_g`, `f_proj_1`, `g_proj_1`, `w_b`; attention `w_out`; router; `latent_down_proj`; shared `w_up_gate`; MoE blocks' shared `w_down`; `lm_head.w_out`), plus embeddings and all vectors |
| 1 | 4 | 4.4e-3 | KDA `f_proj_2`, `g_proj_2` (fan-in 256) |
| 2 | 0.5 | 5.5e-4 | KDA `w_out` (fan-in 2048) |
| 3 | 0.125 | 1.375e-4 | block 0 shared `w_down` (fan-in 8192) |
| 4 | 2 | 2.2e-3 | `latent_up_proj` (fan-in 512) |
| 5 | 2 | 2.2e-3 | routed `w_up_gate` (fan-in 512; from the routed-experts override group) |
| 6 | 1 | 1.1e-3 | routed `w_down` (fan-in 1024; from the routed-experts override group) |

Groups 0-4 come from the default group and 5-6 from the `*routed_experts.*` override. Groups 4
and 5 have the same multiplier but stay separate because they come from different original
groups. The `embeddings.weight` override matches nothing under DDP (pre-existing; see
`PARAMETERIZATION_HANDOFF.md` Section 2.6), so embeddings sit in group 0 with weight decay 0.1.

Medium hero (d=1536, 16 heads), same `base_width`:

| Parameter | m |
|---|---|
| `d -> *` matrices (incl. router, `lm_head`, `latent_down_proj`, shared `w_up_gate`, routed `w_down`) | 0.667 |
| KDA `w_out` (fan-in 4096) | 0.25 |
| attention `w_out` (fan-in 2048) | 0.5 |
| `latent_up_proj`, routed `w_up_gate` (fan-in 768) | 1.333 |
| KDA `f_proj_2`, `g_proj_2` (fan-in 256, does not grow) | 4.0 |
| block 0 shared `w_down` (fan-in 12288) | 0.083 |

---

## 5. Verification performed

- **New tests:** 30 pass: 18 registry tests in `src/test/nn/transformer/mup_test.py`, 11 in
  `src/test/optim/mup_test.py`, and 1 in `src/test/optim/moe_optimizer_test.py`. The FLA-gated
  ones ran on this machine.
- **Regression suites:** `src/test/optim`, `src/test/nn/transformer`, `src/test/nn/ddp` and
  `src/test/train/train_module` gave 288 passed, 85 skipped and 3 failed. All 3 failures
  reproduce identically on a clean `git archive HEAD` copy, which has no `optim/mup.py`:
  - `model_test.py::test_small_ngpt_builder_config` (2 cases): `NormalizedAttention.__init__()
    got an unexpected keyword argument 'scalable_softmax'`.
  - `headwise_qk_checkpoint_test.py::test_distributed_checkpoint_roundtrip`: a spawned process
    exits with code 1.
- **Real-hero group audit** (one B300, `/tmp/mup_audit/hero_groups.py`, **not checked in**):
  - Setup: single-process NCCL (`world_size=1`); real small hero (512 experts, nanochat init,
    per-head QK gains, `use_cute_kernel=False` since `kernel_fun` isn't installed); wrapped in
    `MultiGroupDistributedDataParallel(model, init_sync=False)`;
    `OLMoDDPOptimizerConfig.build_groups` run with the hero's overrides plus
    `mup=MuPConfig(base_width=1024)`.
  - Results: all 415 trainable parameters land in exactly one group, every name has the
    `module.` prefix, and every group's `lr` equals 1.1e-3 times a multiplier derived
    independently from the tensor layout. The script also printed the medium multipliers above
    (meta device).
- **Coordinate check** (`coord_check.py` defaults): per-coordinate RMS change after 10 steps.

```text
                     w=128     w=256     w=512    w=1024   width exponent
standard  embedding  0.01342   0.01223   0.01177  0.01146  -0.07
          middle     0.3336    1.914     10.21    40.69    +2.32
          logits     0.898     1.589     2.352    2.78     +0.55
muP       embedding  0.02181   0.02172   0.02083  0.0194   -0.06
          middle     1.055     1.031     0.9988   0.9822   -0.04
          logits     1.046     1.024     1.003    1.017    -0.02
```

- **Static checks:** isort, black and ruff are clean on all new and changed files. mypy reports
  only the pre-existing error at `src/olmo_core/nn/moe/v2/routed_experts.py:940`.

### 5.1 Scaled weight decay (`scale_weight_decay=True`)

Verified later, on the scaling-ladders mainline ladder (base-model mode pinned to its 275M rung,
`weight_decay=0.1`, embeddings 0.0). The checks measure the decay each parameter actually
receives rather than reading `weight_decay` from the groups. Full write-up:
`ladders/mainline/MUP_WEIGHT_DECAY.md` in scaling-ladders.

- **Unit tests:** the four `scale_weight_decay` tests in section 3.6 pass (75 muP tests in total
  across `src/test/optim/mup_test.py`, `src/test/nn/transformer/mup_test.py` and
  `src/test/optim/completedp_test.py`).
- **GPU smoke runs** (one B300, script `/tmp/mup_smoke/mainline_gpu_smoke_wd.py`, **not checked
  in**): real mainline pretraining config at 275M, 450M, 810M, 1.4B and 2.7B (HSDP, compile,
  `SkipStepAdamW` with foreach, KDA with FLA kernels), with the flag switched on through the CLI
  override `--train_module.optim.mup.scale_weight_decay=true`.
  - Step 1 zeroes every gradient before the optimizer step, so each weight should become exactly
    `fp32(w * fp32(1 - fp32(lr_l * wd_l)))`. Compared against that value computed from
    hand-derived `lr_l * wd_l`, zero elements differed at every rung (all 2.6B parameters at
    2.7B).
  - Steps 2 to 5 fit each weight change to `alpha * w_prev + beta * m_hat / (sqrt(v_hat) + eps)`.
    `-beta / lr` matched `m` (1.0 at 275M down to 0.4167 at 2.7B), and `-alpha / (lr * wd)` was
    0.9999 to 1.0003 for every matrix at every rung. Fit residuals were at most 1e-5.
  - Stored groups had `weight_decay = 0.1 / m` at every rung, with zero mismatches (e.g. 0.24 at
    `m = 0.4167` and 0.2 at `m = 0.5` at 2.7B).
  - **Control:** with the flag off, the same probe at 1.4B measured `-alpha / (lr * wd) = 0.5 = m`
    for every scaled matrix.
  - **Checkpoint resume** (generic `OptimConfig` path, 1.4B): saving at step 3 and resuming with
    the flag on kept `weight_decay = 0.1 / m` and decay ratios of 1.0. Resuming the same
    checkpoint with the flag off kept the checkpoint's `weight_decay = 0.2`; see gotcha 5.
- **Measurement notes.** `SkipStepAdamW` computes its decay factor `1 - step_factor * (lr * wd)`
  in float32, so a measured step-1 decay reads about 0.9999 of the exact value under either
  setting. Zero-initialized parameters (KDA `dt_bias`, `g_proj_2.bias`) show decay only from
  step 3. `lm_head.norm.weight` was excluded from the step-2 fit because its weights and Adam
  direction are almost parallel there; it passes at step 1 and steps 3 to 5.
- **Not exercised:** the DDP path (`OLMoDDPOptimizerConfig`) with `scale_weight_decay=True`, and
  `base_width` mode on GPU (unit tests only).

---

## 6. Gotchas and things that may cause problems later

1. **`base_width` must be identical across the models being compared.** Setting it to each
   model's own `d_model` cancels the scaling and reproduces standard parameterization.
2. **New module types or parameters.** A new module whose weights are not `nn.Linear`,
   `nn.Embedding`, depthwise `nn.Conv1d`, `LayerNorm`, or one of the MoE classes makes
   `named_param_shapes` raise (if ndim >= 2) when `mup` is set. Add a rule in
   `_module_param_shapes`, and keep it consistent with the nanochat init. The consistency tests
   in `src/test/nn/transformer/mup_test.py` should be extended to the new module.
3. **A new 1-D parameter that is really a matrix** (e.g. another flattened weight like the
   routers) silently becomes `vector` (m=1). Add an explicit rule for it.
4. **More optimizer groups.** With muP the small hero has 7 groups instead of 2, so the
   `optim/LR (group i)` metric indices change. `olmoe3_hero_mt.py:169-170` asserts that every
   `optim/LR (...)` metric equals one `CosWithWarmup` schedule; it will fail if muP is enabled
   there and must be updated first.
5. **Generic-path checkpoints** (`OptimConfig`): torch saves param groups in the optimizer state.
   - Switching between standard parameterization and muP, or changing which multipliers exist,
     changes the group count or sizes, and `load_state_dict` raises.
   - After a successful load, only `fixed_fields` (default: `initial_lr`) are re-applied from the
     config. So a changed `base_width` or `embedding_lr_multiplier` takes effect, because it
     changes `initial_lr`, but a changed `scale_weight_decay` or override `weight_decay` does
     not: the checkpoint's value wins. This is the existing behavior for every group
     hyperparameter. Confirmed on GPU (section 5.1): a run resumed from a scaled checkpoint with
     `scale_weight_decay=False` keeps decaying at the scaled rate. Switching the setting
     mid-run needs fresh optimizer state, or `weight_decay` added to `fixed_fields` (untested).
   - **The DDP path** (`OLMoDDPOptimizer`) saves no group hyperparameters, so on resume every
     setting comes from the config.
6. **Weight decay is coupled by default.** Halving a layer's LR also halves its effective decay
   per step. Use `scale_weight_decay=True` to keep `lr * wd` fixed (verified on GPU on the
   generic path, section 5.1). On the DDP path, the scaled
   value is written into every split group (including the default group), which overrides the
   optimizer's default `weight_decay` for those groups.
7. **Tied embeddings.** The shared weight is classified as an embedding (its first owner), so it
   gets `embedding_lr_multiplier`, not the LM-head rule.
8. **The init warning is heuristic and fires once per process** (`log_once` is `lru_cache`d). It
   inspects the first module with an `InitMethod`-typed `init_method` attribute; models without
   one get no warning. The optimizer cannot enforce the init, so set `init_method=nanochat` on
   the model config yourself.
9. **`OptimConfig` subclasses without an `lr` attribute** raise when `mup` is set
   (`_apply_mup`). `scale_weight_decay` additionally needs a `weight_decay` attribute or override.
10. **Global gradient clipping** (`max_grad_norm=1.0` in the hero) clips on the total gradient
    norm, which changes with width and can interfere with LR transfer. Not addressed.
11. **Adam epsilon** stays fixed at the config value; muP strictly wants it to shrink with width.
    Rarely matters at these sizes.
12. **Depth.** muP is width-only, and small -> medium hero also changes depth (16 -> 24 layers).
    Depth transfer would need depth-muP / CompleteP-style residual scaling.
13. **Attention scale.** muP's `1/head_dim` attention scaling is unnecessary because the hero fixes
    `head_dim = 128` and scales the number of heads. If `head_dim` ever scales, `softmax_scale`
    must become an `AttentionConfig` field (it is only an `Attention.__init__` argument today).
14. **Not exercised:** expert parallelism (the `ep_dp` partition is split by the same code but was
    not run with EP > 1), FP8 weight stores, Lion, and `compile=True` in an actual training step.
15. **The coordinate check uses the generic pre-norm `llama_like` model**, not the peri-norm DDP
    hero. It validates the LR mechanism and scaling; the hero path was validated by the
    group-level audit only, not by training.

---

## 7. Possible follow-ups

- An LR-transfer sweep on Beaker at fixed depth (e.g. hero geometry at two widths with the same
  `base_width` and `lr`), which is the real test of muP.
- Update `olmoe3_hero_mt.py`'s LR audit to accept per-group multipliers before using muP there.
- Fix the hero's `embeddings.weight` override pattern (needs a leading `*` under DDP).
- Optional: width-aware gradient clipping, Adam-epsilon scaling, depth-muP.
- Check in the hero group-audit script if it will be needed again.
