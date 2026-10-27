# Muon backends

Both dense and OLMoDDP MoE training support an explicit `muon_backend` choice:

```python
# Dense FSDP/HSDP training, with complete matrix/AdamW routing:
SharedMuonConfig(muon_backend="dion", lr=0.003)

# MoE OLMoDDP training:
OLMoDDPOptimizerConfig(
    algorithm="muon_adamw", muon_backend="dion", compile=False,
    lr=0.003, betas=(0.9, 0.95), weight_decay=0.1,
)
```

Set `muon_backend="shared"` to select the existing reference implementation.
It remains the default. OLMoDDP's default algorithm remains AdamW; choosing a
Muon backend does not enable Muon unless `algorithm="muon_adamw"` is also set.
For either adapter the CLI override is `--train_module.optim.muon_backend=dion`.
The legacy Dion-backed `MuonConfig` remains available and unchanged; use
`SharedMuonConfig` for this common backend selector and skip/checkpoint contract.

## Numerical recipes

`shared` uses FP32 normalization before BF16 matrix multiplies, five fixed
Newton–Schulz iterations, and the existing masked update equations. `dion` uses
Dion's compiled momentum, BF16 direction transport, Polar Express, native RMS
scaling, and batched parameter updates. These are different numerical recipes;
parity between them is not expected. Dion requires RMS scale 0.2. Both use the
configured momentum with Nesterov enabled. Auxiliary parameters keep the
adapter's AdamW implementation and parameter routing, including convolution
parameters and embeddings.

Both adapters keep KDA `w_q`, `w_k`, `w_v`, `w_out`, `f_proj_1/2`, and
`g_proj_1/2` weights on Muon. The per-head scalar write-gate projection `w_b`
uses AdamW, as do all biases, normalization gains (including 2D per-head Q/K
gains), convolution filters, `A_log`, and `dt_bias`. Global-attention gate
projection weights remain on Muon. This routing is independent of backend.
The `w_b` choice follows single-seed MoE hybrid Cx1 retrieval evidence; its
benefit has not yet been validated on the dense ladder. Earlier profiling
runs used Muon for `w_b`. Start fresh optimizer state when changing routing.

Dion's replicated and matrix-sharded paths can also produce different rounding
because their batching and padding differ. They share a numerical recipe, but
do not promise bitwise equality across execution layouts. Validate each adapter
against native Dion with the corresponding layout.

Install Dion through OLMo-core's pinned `dion` extra with the repository's uv
source configuration. The dependency is imported only when selected. Do not
substitute an arbitrary Dion release: this bridge uses the pinned async task API.
Dion compiles its own kernels; leave outer optimizer `compile=False`.

## Execution and state

Dense DTensors use Dion's matrix-sharded all-to-all and same-shape batching.
A single sharded mesh dimension is supported, including HSDP with additional
replicated dimensions. Multiple matrix-sharding dimensions and partial DTensors
are rejected explicitly.

OLMoDDP retains its flat DP/EP-DP state and checkpoint layout. For one weight at a
time, it gathers parameters, gradients, and momentum, then exposes writable
logical matrices to Dion: flattened router matrices, independent routed experts,
and independently split shared/routed up/gate projections. Dion batches these
matrices and distributes their orthogonalization over the one-dimensional state
mesh. It never communicates across the expert model-parallel mesh. For
multi-dimensional state meshes it performs the gathered computation locally.
Updated parameters and momentum are copied back into local flat shards. This
preserves bounded per-weight staging, but the initial FP32 gathers remain; this
is not yet a gather-free MoE execution path.

Batches are bounded by a 256 MiB workspace estimate; a single large matrix may
exceed that estimate. Dion's async runtime can have up to three tasks in flight.
Persistent parameters and momentum use FP32.

Both adapters retain the existing loss/gradient skip decision. Dion skips the
entire optimizer step, preserving parameters, decay, momentum, and counters.
This implementation synchronizes the scalar skip decision to the host once per
step. The shared backend retains its existing device-side masked update.

Checkpoints store the backend choice. Loading a checkpoint into the other backend
raises instead of silently changing the optimizer recipe. Legacy checkpoints
without the field are interpreted as `shared`. The dense adapter also preserves
skip history through PyTorch distributed checkpointing. Backend switching should
start a new optimizer recipe with fresh optimizer state.
