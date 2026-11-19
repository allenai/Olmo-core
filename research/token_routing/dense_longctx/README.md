# dense_longctx: dense full-attention training at 1M tokens (Qwen3.5 hybrid)

Feasibility / engineering thread (2026-11-18/19, experiment `dense_1m`, branch `claude/dense-1m-pipeline`):
can a dense step (every token attends over its full causal context, every token gets gradient) run on
1M-token rows, on how many GPUs, at what cost. Small-model iteration on Lambda A100-80GB.

## 1. The ">262k forward breaks" bug: int32 offset overflow in fla's chunked delta rule (FIXED)

**Cause.** fla 0.4.1 `chunk_gated_delta_rule` stores and reads its per-64-token chunk states at
`h + i_t * H*K*V` (and `dh + i_t*H*K*V` in backward) with a 32-bit loop index
(`fla/ops/common/chunk_delta_h.py`, lines 110-119 and 322-331). The offset wraps once
`(T/64) * H * K * V >= 2^31`, i.e. **T >= 2^30 / (H_v * 128)**:

| model | GDN value heads seen by the kernel | first wrong token |
|---|---|---|
| Qwen3.5-4B / 9B, no CP | 32 | 262,144 |
| Qwen3.5-0.8B / 2B, no CP | 16 | 524,288 |
| Ulysses CP degree c | H_v / c | x c (4B CP4: 1,048,576) |

**Evidence** (`test_int32_overflow.py`, A100, fla 0.4.1, triton 3.5.1). Full-row call vs the same op on exact
state-carrying segments of 131,072 / 65,536 tokens: bit-identical below the bound for every length tested; above it
the forward output differs **from exactly token 262,144 (H=32) / 524,288 (H=16)** on, and once `T*H*128 >= 2^31` from
token 64 on (the wrap corrupts the early chunks too). The backward (dq, dk, dv, dg, dbeta) is wrong from token 0 once
the bound is crossed (H=16, T=600k), identical below it. Outputs stay finite: the failure is silent (wrong CE and
gradients), later NaN / illegal memory access (the 589k crash in `records/grad-sparsity-vs-length.md`). The short
causal conv (checked to 1.5 x 2^31 elements, against torch's conv1d) and the gated output RMSNorm (to 1.1 x 2^31) are
NOT affected. On a real 1,039,552-token lc1m row, Qwen3.5-0.8B base, forward-only answer CE (eval_size = 1 row,
37 answer tokens): **9.06 without the guard, 1.97 with it**.

**Why the October probe (`ops exp show gdn_int32_boundary`, `debug/gdn_int32_boundary/probe.py`, commit dd0828613)
"refuted" it.** It ran one call at T = 262,080 / 262,144 / 262,208 / 272,000 and checked only that the call returned
and `torch.isfinite(o).all()`. The overflow does not crash or produce NaN at those lengths; it silently writes / reads
the wrong chunk states. Comparing against a segmented (or two-call, state-carrying) reference shows it immediately.

**Fix (default off, exact).** `olmo_core.nn.ffn_chunk.enable_int32_safe_long_seq(model)`; trainer flag
`--gdn-int32-safe`; bench flag `--gdn-safe`. Each GatedDeltaNet runs the chunk kernel over segments shorter than
`2^30 / (H_v * D_v)` tokens (a multiple of 1,024), carrying the recurrent state (training and eval). Rows below the
bound take the old path bit for bit. The short conv and output norm get the same segmentation at the 2^31 bound as a
precaution. Module-level exactness at 16k with forced tiny segments: identical outputs, dx rel diff 7e-5. A one-line
upstream fix (`i_t.to(tl.int64)` in those offsets) would remove the need for it; the shared Lambda env was not patched.

## 2. CP cap: not real

Ulysses attention already replicates KV heads when the CP degree exceeds `n_kv_heads`
(`olmo_core/distributed/parallel/context_parallel.py: all_to_all_qkv_cp2hp`); the constraint is `cp | n_heads`
(Qwen3.5-4B / 9B: 16 -> CP up to 16; 0.8B / 2B: 8). GatedDeltaNet needs `cp | n_v_heads` and `cp | key_dim` (16 heads:
up to 16). The "qwen3_5 max CP 4" was only the trainer's help text (corrected). FSDP shards parameters over dp x cp.

## 3. Measured dense steps (Lambda A100-80GB, 2026-11-19)

One step = forward over the whole row + backward + AdamW, batch = 1 row per CP group, full activation
checkpointing (every block), bf16 params / fp32 reduce (FSDP over dp x cp), fused-linear CE, flash-attn 2.8.3,
int32 guard on. Synthetic rows (random tokens, 40 supervised tokens) with random init unless "real" (Qwen3.5 base
weights + the first lc1m held-out row, 1,039,616 tokens, 37 answer tokens). PF/row = analytic training FLOPs
(2 x MACs; linears 6NT full FT / 4NT LoRA, causal attention 6 T^2 H D per layer, LM head on every position).
s/step = mean of steps 2-3 (step 1 includes kernel autotune). GPU-s/row = s/step x GPUs.

| model | T | GPUs | CP | mode | peak GiB alloc (reserved) | s/step | GPU-s/row | PF/row | extra options |
|---|---|---|---|---|---|---|---|---|---|
| 0.8B | 64k | 1 | 1 | full FT | 22.7 (25.9) | 6.3 | 6 | 0.6 | |
| 0.8B | 131k | 1 | 1 | full FT | 32.2 (35.7) | 18.5 | 18 | 1.9 | |
| 0.8B | 262k | 1 | 1 | full FT | 51.3 (55.7) | 62.8 | 63 | 6.3 | |
| 0.8B | 262k | 4 | 4 | full FT | 14.2 (15.9) | 16.9 | 67 | 6.3 | |
| 0.8B | 524k | 1 | 1 | full FT / LoRA | OOM | | | | |
| 0.8B | 524k | 2 | 2 | full FT | 46.2 (50.0) | 116.9 | 234 | 22.7 | |
| 0.8B | 524k | 4 | 4 | full FT | 24.4 (26.7) | 58.9 | 236 | 22.7 | |
| 0.8B | 1M | 1 | 1 | full FT | OOM | | | | |
| 0.8B | 1M | 2 | 2 | full FT / LoRA | OOM (~75 GiB, then a 3.5 GiB T/CP x d_ff FFN buffer) | | | | |
| 0.8B | 1M | 2 | 2 | LoRA r16 | 74.1 (75.8) | 449 | 899 | 84.3 | FFN chunked (65,536-token chunks); from the log, the case hit the 25-min sweep timeout after step 2 |
| 0.8B | 1M | 2 | 2 | full FT | step 1 74.1, OOM at step 2 (Adam state) | | | | FFN chunked |
| 0.8B | 1M | 2 | 2 | full FT | 55.0 (57.9) | 442.6 | 885 | 85.9 | FFN chunked + checkpoints offloaded to pinned CPU memory |
| 0.8B | 1M | 4 | 4 | full FT | 44.7 (48.9) | 220.0 | 880 | 85.9 | |
| 0.8B | 1M | 4 | 4 | LoRA r16 | 41.1 (43.9) | 222.3 | 889 | 84.3 | |
| 4B | 262k | 8 | 8 | LoRA r16 | 18.5 (20.8) | 25.1 | 201 | 18.0 | |
| 4B | 524k | 8 | 8 | LoRA r16 | 31.0 (34.1) | 83.9 | 671 | 63.0 | |
| 4B | 1M | 8 | 8 | full FT | 62.3 (68.7) | 305.7 | 2,445 | 242.9 | |
| 4B | 1.04M real | 8 | 8 | LoRA r16 | 55.5 (60.6) | 305.3 | 2,442 | 230.4 | base weights + real lc1m row; loss 1.784 -> 1.452 (steps 1-2) |

Throughput is flat at ~95-100 TFLOP/s per A100 (~31% of bf16 peak) from 262k up: at 1M, 89-94% of the FLOPs are the
full-attention layers (flash-attn, O(T) memory), so LoRA saves nothing in time and little in memory (the activation
wall, not the optimizer state, sets the GPU count). CP does not change GPU-s per row (Ulysses scales cleanly here).

**Where the memory goes (per GPU, T_local = T / CP).** (a) the activation-checkpoint block inputs, n_layers x T_local x
d_model x 2 B (0.8B at 1M / CP2: 24 GiB; 4B at 1M / CP8: 20 GiB); (b) one block's recompute + backward over T_local:
the FFN intermediates (T_local x d_ff x 2 B per buffer, several live: 3.5 GiB each for 0.8B at CP2) dominate, then the
Ulysses all-to-all buffers of the attention layers (the full T for H / CP heads); (c) params / grads / Adam, sharded
over all GPUs (small: 4B full FT is ~8 GiB per GPU over 8). The GDN chunk states (T/64 x H x 128 x 128) are
segmented by the guard. Options that moved the wall, all exact: CP (divides (a) and (b)); FFN sequence chunking
(`--ffn-chunk-tokens`, removes most of (b)); offloading (a) to pinned CPU memory (`torch.autograd.graph.save_on_cpu`,
-23 GiB at 0.8B / 1M / CP2 with no measurable slowdown, A100 nodes have ~1 TB RAM).

**Real-row checks of the guard** (base weights, forward-only answer CE of one lc1m row: eval_size = 1 row, a
correctness check, not a metric): 0.8B at 1,039,552 tokens no CP: 9.06 off / 1.97 on; 4B at 400,000 tokens no CP:
NaN off / 1.84 on; 4B at 1,039,616 tokens CP2 (bound 524,288): 8.87 off / 1.79 on; 4B at 250,112 tokens (below the
bound): 1.88. The 4B LoRA training step at CP8 on the same 1.04M row starts at loss 1.784, matching the CP2 eval
forward (1.785).

## 4. Extrapolation to Qwen3.5-4B dense training at 1M

Measured: one 8x A100-80GB node runs a dense 4B step at 1M (full FT 62 GiB per GPU, LoRA lower), 306 s per row =
0.68 A100-hours per row. Extrapolated (not measured):
- **H100-80GB**: same memory (fits on 8 with CP8; CP4 on 4 H100s would need ~2x the per-GPU activations, ~100+ GiB:
  no). Flash-attn 2 on H100 runs ~1.5-2x A100 at head_dim 256 (FA3 would be more): ~150-200 s per row on 8 H100s,
  0.35-0.45 H100-hours per row.
- **16 GPUs (CP16, allowed: n_heads = 16)**: halves the per-GPU activations (headroom for 2M or for no offload /
  less recompute); time per row ~halves (Ulysses all-to-all volume per GPU is constant), GPU-hours per row unchanged.
- A 360-row lc1m epoch at 0.68 A100-h/row = ~245 A100-hours (~31 h on one node); 4 rows per step with gradient
  accumulation is just 4 sequential rows.
- The trainer path (train_ctc_suite, `--cp-degree 8 --gdn-int32-safe --activation-checkpointing full --skip-dry-run`) is
  what section 5 checks; what remains to build: see section 6.

## 5. The real trainer at 1M (train_ctc_suite on the lc1m shard), measured

Lambda job 115320 (4B base, contradiction lc1m shard, 1,048,576-token rows, 8 A100s, 3 steps, LoRA r16, no save):

    cd $L/recipes/train/research_trainer && sbatch --partition=lambda --account=site --qos=preemptive_high \
      --gres=gpu:A100:8 --nodes=1 --cpus-per-task=64 --mem=600G --time=01:30:00 --job-name=d1m-trainer-4b-r3 \
      --export=ALL,REPO=$L,TASK=contradiction,DATA_SRC=lc1m,VARIANT=full,SCALE=4b,RUN=d1m-trainer-4b-cp8-lora-r3,\
    NGPU=8,SHARD_DEGREE=1,SEQ_LEN=1048576,EPOCHS=1,GLOBAL_BATCH=1,MICRO_BATCH=1,MAX_STEPS=3,LR=1e-5,ACT_CKPT=full,\
    ATTN_BACKEND=flash_2,BASE_SRC=$LROOT/bases/q35-4b-base-modelonly,SAVE_ROOT=/tmp/prasann/d1m/ckpts,\
    WORK_ROOT=/tmp/prasann/d1m,SAVE_CKPT_FLAG=,EXTRA='--lora-rank 16 --lora-scale 32 --cp-degree 8 --gdn-int32-safe --skip-dry-run --ffn-chunk-tokens 65536' \
      run_ctc_lambda.sbatch

| step | train CE | grad norm | GPU active / reserved GiB | TFLOP/s per GPU (meter) | MFU | s/step |
|---|---|---|---|---|---|---|
| 1 | 1.806 | 72.2 | 48.3 / 52.1 | | | (incl. warmup) |
| 2 | 1.896 | 74.8 | 48.9 / 52.7 | 99.0 | 31.7% | 307 (427 tokens/s/GPU) |
| 3 | 1.726 | 55.0 | 48.9 / - | 99.2 | 31.8% | 306 (428 tokens/s/GPU) |

The trainer's FLOP meter (486 PF after 2 steps = 243 PF/row) agrees with the analytic count. Two launch traps hit on
the way: (1) `run_ctc_lambda.sbatch` defaults `SHARD_DEGREE=$NGPU`, which the CP mesh rejects (dp world = NGPU/CP = 1):
pass `SHARD_DEGREE=1` (FSDP still shards over dp x cp = 8). (2) Without `--ffn-chunk-tokens` the trainer step used
71 GiB at step 1 and OOMed in step 2 (FFN + LoRA recompute), 16 GiB more than the bench for the same config; with
65,536-token FFN chunks it runs at 49 GiB. The 16 GiB trainer-vs-bench gap was not tracked down.

## 6. Verdict and what is left

**Dense 1M training of Qwen3.5-4B is feasible today on one 8x A100-80GB node** (Ulysses CP8, full AC, int32 guard,
FFN chunking in the trainer): ~306 s and 0.68 A100-hours per 1M-token row, full FT or LoRA, 49-62 GiB per GPU.
The pipeline exists end to end in train_ctc_suite (flags above). Still to build / decide:
- the int32 guard should be ON for every long-row Qwen3.5 run (or fla patched upstream: `i_t` -> int64 in
  `chunk_delta_h.py`); it is default-off only to leave existing behaviour bit-identical;
- the launcher's SHARD_DEGREE default under CP (one-line fix, not made here);
- the dense held-out eval at 1M uses the same forward (CP or no CP + guard) and is affordable (4B 1M forward on 2 A100s:
  ~225 s);
- not run: CP16 / two nodes, H100s, 2M rows, throughput tuning (FA3, compile, selective AC).

## Files

| file | what |
|---|---|
| `test_int32_overflow.py` | localises the overflow per fla op (conv / chunk fwd / chunk bwd / output norm) and checks the guard on the olmo_core GDN module |
| `dense_step_bench.py` | one dense training step (fwd + bwd + AdamW) through the trainer's own model build / FSDP(HSDP) / Ulysses CP / AC path, synthetic or real (`heldout_lc.npz`) rows; JSON per run |
| `sweep.sh`, `sweeps/*.txt` | back-to-back bench configs inside one allocation |
| `run_lambda.sbatch`, `stage_lambda.sh` | Lambda payload + code staging (rsync to `/accounts/projects/sewonm/prasann/ctc_suite/OLMo-core-dense1m`) |
| `collect.py` | tabulates result JSONs |
| `data/bench/*.json` | the measured runs |
