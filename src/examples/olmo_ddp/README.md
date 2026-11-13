# Olmo-core DDP MoE examples

This directory holds the training entry points used for the Olmo-core MoE tech
report, *Supercharging Olmo-core for Efficient and Scalable MoE Training*. Every
script here builds against the `OLMoDDPModel` stack in this revision of
Olmo-core. Scripts that were written against the pre-merge `moe-v2-core` API
have been ported to the current API without changing their experiment settings
(see [Porting notes](#porting-notes)).

## Launching

All scripts use the same generic entry point:

```bash
torchrun --nproc-per-node 8 [--nnodes N --rdzv-endpoint HOST:PORT] \
  src/examples/olmo_ddp/<script>.py RUN_NAME [OVERRIDES...]
```

or, on Beaker, `python -m olmo_core.launch.beaker ... -- <script>.py RUN_NAME`.

Requirements shared by all scripts:

- the `beaker` extra (`olmo_core.internal` imports it), plus `flash-attn-4` for
  the FlashAttention 4 backend;
- a writable `/workspace` (checkpoints, work dir) and read access to the Olmo
  data mix at `s3://ai2-llm` (`DataMix.OLMo_mix_0925`);
- for scripts on the DeepEP v2 path, a DeepEP checkout at `OLMO_DEEPEP_PATH`
  (default `/workspace/DeepEP`).

The production scripts assume eight GPUs per node; the node count is
`REF_NUM_NODES` in each script.

## Production throughput runs

These are the operating points reported in the production-throughput section
and the run tables of the report's experiment appendix. All use sequence length
8192, top-4 routing, one shared expert, and B300 GPUs. Active and total
parameter counts below are what each script's model config reports; they match
the report.

| Report operating point | Script | Nodes | EP / PP | GBS | Routing | Precision, recompute | Active @ total |
|---|---|---:|---:|---:|---|---|---:|
| Tiny-64E, no EP (headline) | `OLMoE3-dev-t001-random.py` | 2 | 1 / 1 | 8 Mi | random | BF16, none | 1.59B @ 12.91B |
| Tiny-64E, no EP (learned companion) | `OLMoE3-dev-t001.py` | 2 | 1 / 1 | 8 Mi | learned | BF16, none | 1.59B @ 12.91B |
| Tiny-64E, EP8 (learned companion) | `OLMoE3-dev-t002.py` | 2 | 8 / 1 | 8 Mi | learned | BF16, none | 1.59B @ 12.91B |
| Small-64E, 24 Mi (headline) | `OLMoE3-dev-s001-random.py` | 16 | 8 / 1 | 24 Mi | random | BF16, none | 4.29B @ 40.86B |
| Small-64E, 24 Mi (learned companion) | `OLMoE3-dev-s001.py` | 16 | 8 / 1 | 24 Mi | learned | BF16, none | 4.29B @ 40.86B |
| Small-64E, 16 Mi (alternate) | `OLMoE3-dev-s001-random-16mi.py` | 16 | 8 / 1 | 16 Mi | random | BF16, none | 4.29B @ 40.86B |
| Medium-64E (headline) | `OLMoE3-dev-m001-64e-16mi.py` | 16 | 8 / 2 | 16 Mi | random | BF16, none | 7.37B @ 70.22B |
| Medium-96E (headline) | `OLMoE3-dev-m001.py` | 16 | 8 / 2 | 24 Mi | random | BF16, none | 7.38B @ 103.75B |
| Medium-128E (headline) | `OLMoE3-dev-m002.py` | 16 | 8 / 4 | 24 Mi | random | BF16, none | 7.38B @ 137.27B |
| Large-128E, PP4 (headline) | `OLMoE3-dev-l001.py` | 64 | 8 / 4 | 32 Mi | random | BF16, none | 15.14B @ 295.99B |
| Large-128E, PP8 (alternate) | `OLMoE3-dev-l001-pp8-24mi.py` | 64 | 8 / 8 | 24 Mi | random | BF16, none | 15.14B @ 295.99B |
| Ultra-128E (headline) | `OLMoE3-dev-u001.py` | 64 | 8 / 8 | 64 Mi | random | MXFP8, per-layer | 58.36B @ 1.200T |
| Ultra-256E (headline) | `OLMoE3-dev-u002.py` | 64 | 32 / 8 | 64 Mi | random | MXFP8, per-layer | 58.41B @ 2.380T |

Notes:

- Files with a suffix (`-random`, `-16mi`, `-64e-16mi`, `-pp8-24mi`) are exact
  copies of their base script with only the constants named in their first two
  comment lines changed. They exist so that every reported operating point has
  its own entry point.
- Ultra-128E uses rowwise NVSHMEM EP. Ultra-256E uses DeepEP v2 because its EP32
  groups span nodes; `u002` selects `ExpertParallelPath.deepep_v2` in
  `build_model_config` even though `USE_ROWWISE_A2A = True` (that flag enables the
  no-sync block path).
- Pipeline-parallel scripts (`MINUS_LAST_STAGE = 1`) use two dense leading blocks;
  Tiny and Small use one. The parameter counts above include this.
- `OLMoE3-dev-t001.py`, `OLMoE3-dev-t001-random.py`, and `OLMoE3-dev-t002.py` set
  `USE_NV_PROFILE = True`, which enables NVTX ranges and an Nsight capture window.
  Set it to `False` for throughput measurements outside the profiled window.

## Other experiments in the report

| Experiment | Scripts | Controls |
|---|---|---|
| Matched eight-layer DDP+EP vs FSDP/HSDP benchmark (capacity and global-batch sweeps) | `moe_8l_ddp.py`, `moe_8l_fsdp.py`, `dense_8l_ddp.py`, shared settings in `moe_8l_common.py` | `TECH_REPORT_NUM_EXPERTS`, `TECH_REPORT_GLOBAL_BATCH_SIZE`, `TECH_REPORT_PARALLEL_DEGREE` (8 for EP8, 1 for no-EP DDP) |
| Four-GPU MXFP8 recipe ablation and per-block recompute benchmark | `moe_8l_ddp.py` | `TECH_REPORT_MXFP8_MLP`, `TECH_REPORT_MXFP8_ATTN_QKV`, `TECH_REPORT_MXFP8_ATTN_OUT`, `TECH_REPORT_MXFP8_ATTN_SAVE_QKV`, `TECH_REPORT_RECOMPUTE_EACH_BLOCK`, `TECH_REPORT_EP_PATH` |
| EP transport profiles (synchronized 1D, rowwise NVSHMEM, DeepEP v2; with and without shared experts; no-EP control; six-layer MXFP8 variants) | `moe_8l_ep8_{sync,rowwise,deepep_v2}_profile.py`, `moe_8l_ep8_{sync,rowwise,deepep_v2}_no_shared_profile.py`, `moe_8l_ep1_no_shared_profile.py`, `moe_6l_ep8_{rowwise,deepep_v2}_mxfp8_no_shared_profile.py` | `TECH_REPORT_RANK_MICROBATCH_SEQUENCES` (2 or 4) |
| Load-balancing-loss weight runs (Token Gerrymandering) | `moe_8l_ddp_lbl_0p02.py`, `moe_8l_ddp_lbl_0p20.py`, `moe_10l_ddp_lbl_0p05.py`, `moe_10l_ddp_lbl_0p50.py`, `moe_10l_ddp_lbl_0p50_shared.py` | `TECH_REPORT_DATA_ROOT`, `TECH_REPORT_SAVE_ROOT` |

**Two-batch overlap is not available in this revision.** The core
`ExpertParallelConfig` rejects the `tbo` schedule, and `moe_8l_ddp.py` raises a
clear error for `TECH_REPORT_TWO_BATCH_OVERLAP=1`. The report's two-batch-overlap
measurement was taken on the pre-merge branch and cannot be reproduced from here.

### Interpreting the eight-layer DDP vs FSDP comparison

This is the closest supported comparison, not a one-line parallelism toggle.
`OLMoDDPModel` rejects FSDP wrapping and the generic Transformer stack rejects the
Olmo-core DDP train module, so:

- the DDP run uses the fused block, v2 router, Olmo multi-group reducer, Olmo
  optimizer, and rowwise NVSHMEM EP;
- the FSDP/HSDP run uses the generic hybrid-MoE block, v1 router, generic
  optimizer, and FSDP2 wrapping.

The dimensions, active experts, dense first layer, capacity factor, attention
backend, data shape, and measurement window are matched; router details,
shared-branch mixing, optimizer implementation, and sparse kernels are not.
Both MoE entry points build the same parameter count (3.17B active @ 24.31B
total at the default 64 experts). Report the result as a DDP+EP stack versus an
FSDP/HSDP stack, not as an isolated estimate of the DDP versus FSDP API.

## Porting notes

The 16 scripts below were written against the pre-merge API and ported to this
revision: `dense_8l_ddp.py`, `moe_8l_fsdp.py`, the five load-balancing-loss
scripts, and the nine EP transport profile scripts. The port changes only API
surface:

- block configs use `sequence_mixer=` and `layer_norm=` (which builds both the
  attention and feed-forward norms) instead of `attention=`, `attention_norm=`,
  and `feed_forward_norm=`;
- attention width is `n_heads * head_dim`; the removed `d_attn=` argument always
  equalled that product in these scripts;
- the `main(config_builder=...)` entry point is replaced by the explicit
  `CliContext` / `build_config` / `train` block used by the other scripts here,
  so the command line is `SCRIPT RUN_NAME [OVERRIDES...]` instead of
  `SCRIPT train RUN_NAME CLUSTER`.

Each ported script was checked by building its model, train-module, and trainer
configs against both the pre-merge revision and this one, and comparing the
active and total parameter counts, layer count, dense blocks, EP path and
schedule, router settings, global batch, rank microbatch, EP and PP degrees,
data-parallel type, learning rate, weight decay, run duration, callbacks, and
profiler setting. All fields matched. `src/test/examples/olmo_ddp_tech_report_test.py`
builds every script in this directory on CPU and pins the parameter counts above.
