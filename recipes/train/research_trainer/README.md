# recipes/train/research_trainer — the task-agnostic research trainer (`train_ctc_suite.py`)

Formerly `src/scripts/train/memexpress/ctc_suite/`. This is the plain-torchrun SFT trainer that
every research arm (token routing, FFN compute, KV route, landmark — see `research/<family>/`)
drives through its `--variant` and routing flags; the canonical setA dense SFT does **not** use it
(that is `recipes/train/setA/`). Results and checkpoints live outside git (node-local `/data`,
weka `.../prasanns/ctc_suite/ckpts/`).

Extra files vs. the original folder: `beaker_ctc_suite.py` (Beaker launcher), `llama_configs.py` /
`olmo3_configs.py` / `olmo_hybrid_configs.py` (non-Qwen backbones), and `convert_qwen35_base.py` +
`convert_qwen35_bases.sbatch` with `base_conversion/` (vendored `corpus_reasoning.lib.olmo_models`
and `corpus_reasoning.train.convert_hf_to_olmo`) for HF → olmo-core base conversion.

## Original scope: CTC-suite scaling experiment (full vs document-chunked, ~25 tasks)

Training entrypoints for `records/ctc-suite-scaling-plan.md`: for every task in the
corpus-reasoning suite, one **joint** SFT run per (task, arm) on hybrid Qwen3.5 (0.8B/4B/9B),
arms `full` (plain causal) vs `chunked` / `chunked-mix` (document-chunked mask on the
full-attention blocks; GDN blocks stay unmasked). Eval always uses the pure chunked mask for the
chunked arm.

| File | What |
|---|---|
| `train_ctc_suite.py` | Task-agnostic local SFT trainer (torchrun; generalizes `attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py` + the 0.6B script's curriculum hard-fail) |
| `run_ctc_local.sbatch` | Single-node 8-GPU slurm launcher (self-submits; serves berkeleynlp AND jsteinhardt nodes) |
| `run_ctc_lambda.sbatch` | Single-node 8×A100 slurm launcher for the Berkeley `lambda` partition (no weka/Beaker, no internet — see below) |
| `beaker_ctc_suite.py` | Beaker launcher (4B/9B on jupiter, urgent, minRuntime auto) wrapping the same CLI |
| `convert_qwen35_base.py` / `convert_qwen35_bases.sbatch` | Local HF Qwen3.5 base → olmo-core model-only distcp (+ marker audit) |

Data shards come from `recipes/data/convert_unified_to_document_landmark.py` with
`--marker-set qwen3_5` (box 248049/248050, eos 248044 — the trainer imports these from
`RESERVED_IDS["qwen3_5"]` and hard-fails on a shard built with any other marker set). The
**same shard** feeds both arms; `full` simply treats the markers as ordinary tokens.

Before training a new base scale: marker-embedding audit (cos AND norm), then model-only distcp
conversion — see plan §4 (0.8B audited PASS 2026-07-18; 4B/9B must be audited when converted).

## Trainer knobs (`train_ctc_suite.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--task` | required | Suite task name (provenance + wandb; cross-checked vs shard metadata) |
| `--data` | required | qwen3_5 shard dir (`token_ids_part_*.npy` + `labels_mask_*.npy` + `metadata.json`) |
| `--variant` | required | `full` \| `chunked` \| `chunked-mix` (curriculum mix 0.8→0.0) |
| `--model-scale` | `0.8b` | `0.8b` / `4b` / `9b` → `TransformerConfig.qwen3_5_{0_8B,4B,9B}` |
| `--seq-len` | `40960` | Fits the 32k rung + prompt/CoT overhead; guarded vs shard `max_example_len` |
| `--epochs` | `3` | Plan lever: drop to 1 at 20k examples if wall-clock demands |
| `--lr` | `5e-5` | Inherited from the source attn_explore recipe |
| `--global-batch` | `8` | Instances per optimizer step (tokens = this × seq-len); must be ≥ world_size |
| `--micro-batch-instances` | `1` | Instances per rank per forward (throughput only; exact grad-accum) |
| `--base-checkpoint` | 0.8b default | `model_and_optim` distcp subdir; REQUIRED for 4b/9b; `.metadata` guarded |
| `--save-folder` / `--run-name` | — | Fresh save-folder per run (silent auto-resume trap) |
| `--resume` | off | A new launch into a save folder holding ANOTHER launch's step checkpoints fails (`run_identity.json` records the owning run + Beaker experiment / slurm job); a preemption restart of the same launch resumes without it. Pass this to continue such a checkpoint on purpose |
| `--skip-marker-audit` | off | Rank 0 audits the base's marker rows (`doc_start`/`doc_end`, + `landmark` for sparselandmark) at start with `fix_marker_embeddings.py`'s cos + norm gates and refuses an unrepaired base |
| `--max-tokens` | `0` | Sub-budget point with a full schedule; refused if it buys < 2 optimizer steps (`global_batch × seq_len` tokens each) |
| `--wandb-group` / `--wandb-entity` / `--no-wandb` | group=run name | Project `memory-networks`; group URL printed at launch |
| `--seed` | `0` | Run-to-run seed offset for noise-floor runs |
| `--mix-start-p` / `--mix-end-p` / `--mix-seed` | `0.8` / `0.0` / `42` | chunked-mix curriculum (hard-fails if the anneal cannot land on `mix_end_p`) |
| `--max-steps` / `--save-checkpoint` / `--no-compile` | — | As in the source scripts; `chunked-mix` forces no-compile |
| `--attn-backend` | `flash_2` | `flash_2` \| `torch`; escape hatch for a cluster where flash-attn isn't importable/verified for the GPU arch (SDPA works everywhere but the saved checkpoint then can't do KV-cached eval decoding) |
| `--dry-run` (+ `--dry-run-world-size`, `--dry-run-n-examples`) | — | CPU-only: builds configs + curriculum math, prints the plan, writes nothing |

The curriculum derivation carries BOTH divisor fixes (world_size and micro-batch-instances) and
asserts the predicted final `p_standard` equals `mix_end_p` (±0.01), refusing to start a run that
would silently keep training on plain causal (mask-mix-ngpu-anneal bug lineage).

## sbatch knobs (`run_ctc_local.sbatch`)

Run knobs (env): `TASK DATA_SRC VARIANT SCALE RUN EPOCHS SEQ_LEN LR GLOBAL_BATCH MICRO_BATCH
BASE_SRC WANDB_GROUP MAX_STEPS NGPU SEED`. `BASE_SRC` defaults to the 0.8B base
(`/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly`); pass it explicitly for 4b/9b.

Placement knobs (env, used when the file is **executed directly** — it then self-submits via
sbatch): `PARTITION` (default `berkeleynlp`), `QOS` (optional), `ACCOUNT` (optional), `NODE`
(optional `-w`), `TIME` (default `06:00:00`). jsteinhardt nodes (mooney/cubbins/mcfuzz/sneetches)
need `PARTITION=jsteinhardt QOS=preemptive_high ACCOUNT=site`. Plain `sbatch` also works —
placement then comes from sbatch CLI flags, which override the header.

NFS-safety is inherited from the attn_explore launcher: log to node-local `/data` (mirrored to
`/scratch/users/prasann/ctc_suite_logs/` every 30 s and on exit), data/workdir/ckpts staged to
`/data`, base read in place from `/scratch`, `.metadata` guard, `PYTHONWARNINGS=ignore`.

## `lambda` (Berkeley SLURM, no weka/Beaker) — `run_ctc_lambda.sbatch`

The Berkeley `lambda` partition (5 nodes × 8×A100-SXM4-80GB, ssh alias `lambda`) has **no
internet** on either the head or compute nodes (github.com, huggingface.co, api.wandb.ai all
unreachable — verified with `curl --max-time 8`) and no weka. Everything is staged over NFS under
`LROOT=/accounts/projects/sewonm/prasann/ctc_suite/` from a host that *does* have access
(code/base/shard/eval rsync'd in directly; no `git clone` on lambda). The nodes do have node-local
NVMe (`/data`, `/tmp`), but per the working precedent in
`corpus-reasoning/jobs/*lambda*.sh` (esp. `stage_lambda_niah.sh`, `lambda_train_niah_cmp.sh`)
everything stays on the LROOT NFS mount — `run_ctc_lambda.sbatch` does not invent a node-local
staging path.

Env: a conda-pack'd relocatable env at `/accounts/projects/sewonm/prasann/olmo_test/env`,
activated with `source $ENV/bin/activate && conda-unpack` (NOT `conda activate`). That env ships
torch 2.9.1+cu128 and flash-attn 2.8.3 (A100 = sm80, both importable and compatible — the known
flash-attn 2.8.3 SIGSEGV bug is H200-specific, not present on A100). It was **missing `fla`**
(flash-linear-attention, required for Qwen3.5's Gated DeltaNet blocks — `has_fla()` asserts and
crashes model construction without it); fixed by rsync'ing the pure-Python/Triton `fla` package
(no compiled extensions) from the horton `corpus-reasoning-olmo` env's site-packages into the
lambda env's site-packages. Re-check `has_fla()` after any env rebuild.

**Critical gotcha**: launcher scripts that `source $ENV/bin/activate` must use `set -o pipefail`
only — **not** `set -u`/`set -uo pipefail`. The conda-pack activate script references unset
internal vars (e.g. `CONDA_PREFIX`) and dies instantly under `set -u` (this is exactly what
`run_ctc_local.sbatch`, which has no such `source activate` step, gets away with).

Knobs mirror `run_ctc_local.sbatch`: `TASK DATA_SRC(a directory NAME under LROOT/data, not a full
path) VARIANT SCALE RUN EPOCHS SEQ_LEN LR GLOBAL_BATCH MICRO_BATCH BASE_SRC ATTN_BACKEND
WANDB_GROUP MAX_STEPS NGPU SEED`, placement via `TIME` / `NODE` when self-submitted. `WANDB_MODE`
is forced to `offline` unconditionally (no internet); sync wandb runs from a host with access
afterward if needed.

```bash
# pilot: contradiction n20 shard (2000 instances, seq 2048), full-attn arm, 1 epoch, 8×A100
ssh lambda
cd /accounts/projects/sewonm/prasann/ctc_suite/OLMo-core/recipes/train/research_trainer
TASK=contradiction DATA_SRC=contradiction_n20_docdense_nocot_qwen35 VARIANT=full SCALE=0.8b \
  EPOCHS=1 SEQ_LEN=2048 RUN=ctc-contra-full-08b-lambda-pilot TIME=00:40:00 \
  ./run_ctc_lambda.sbatch
```

To refresh the staged code/data after local edits (from a host with access to both, e.g. the
horton login host):

```bash
for d in src recipes; do
  rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    $d/ lambda:/accounts/projects/sewonm/prasann/ctc_suite/OLMo-core/$d/
done
```

## Example invocations

CPU sanity first (always cheap, catches geometry/curriculum errors before burning a node):

```bash
PYTHONPATH=src python recipes/train/research_trainer/train_ctc_suite.py \
  --task contradiction --data /scratch/users/prasann/ctc_suite_data/contradiction_joint_20k \
  --variant chunked-mix --dry-run --dry-run-world-size 8
```

**(a) 30-min pilot** (contradiction, ~2.5k-example shard, short ctx — §5 scale selection /
pure-vs-mix pilot; horton):

```bash
TASK=contradiction DATA_SRC=/scratch/users/prasann/ctc_suite_data/contradiction_pilot_2p5k \
  VARIANT=chunked-mix SCALE=0.8b SEQ_LEN=8192 EPOCHS=1 TIME=00:45:00 \
  RUN=ctc-contra-cmix-08b-pilot WANDB_GROUP=ctc-suite-pilot-contradiction \
  ./run_ctc_local.sbatch
# and the pure-chunked pilot arm: VARIANT=chunked RUN=ctc-contra-chunked-08b-pilot ...
```

**(b) Final joint run** (20k examples, n ~ U[n(2k), n(32k)], seq 40960; both arms from the SAME
shard; jsteinhardt node):

```bash
for V in full chunked-mix; do
  PARTITION=jsteinhardt QOS=preemptive_high ACCOUNT=site NODE=mooney TIME=12:00:00 \
  TASK=contradiction DATA_SRC=/scratch/users/prasann/ctc_suite_data/contradiction_joint_20k \
  VARIANT=$V SCALE=0.8b RUN=ctc-contra-$V-08b-joint WANDB_GROUP=ctc-suite-contradiction \
  ./run_ctc_local.sbatch
done
```

**(c) Per-N control runs** (repro-gate T0: contradiction per-N shards, both arms, 0.8B —
isolates the joint-training protocol from pipeline bugs):

```bash
for N in 20 50 100; do for V in full chunked-mix; do
  TASK=contradiction DATA_SRC=/scratch/users/prasann/ctc_suite_data/contradiction_n${N}_ctl \
  VARIANT=$V SCALE=0.8b SEQ_LEN=8192 RUN=ctc-contra-n${N}-$V-08b \
  WANDB_GROUP=ctc-suite-contra-perN ./run_ctc_local.sbatch
done; done
```

## Provenance / wandb conventions

- Every run writes `provenance.json` into its save folder before fitting: git commit, full args,
  data path, marker-id set, world size, and `start_time` (from `CTC_LAUNCH_TS`, which the sbatch
  exports at submit time). Result JSONs must carry this forward (plan §8); never fabricate
  metadata.
- wandb: project `memory-networks`, one group per task (`WANDB_GROUP=ctc-suite-<task>`), run
  name = `ctc-<task>-<variant>-<scale>[-tag]`. The trainer **prints the group URL at launch**
  (set `WANDB_ENTITY` or `--wandb-entity` for a direct link) — always surface it when launching.
- Fresh `--save-folder` per run. A relaunch into a folder that holds another launch's step
  checkpoint now fails at start instead of silently resuming (`--resume` to continue it on
  purpose). `kill -USR1 <rank pid>` prints every thread's stack into the job log.
- Reporting: `n` = corpus size only; eval-set size is `eval_size` (≥500, or flagged inline).
