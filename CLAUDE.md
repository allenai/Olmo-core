# CLAUDE.md

Guidance for Claude Code in this repo: AI2's OLMo-core training library plus our long-context
research (CTC suite, attention / routing experiments). **Upstream OLMo-core architecture, commands,
code style, docstrings, test conventions and Docker/Beaker image mechanics:
[`docs/claude/upstream-olmo-core.md`](docs/claude/upstream-olmo-core.md)** — read it before changing
`src/olmo_core/`. Everything below is project rules.

## Research ops (runs, results, experiment state)

`/accounts/projects/berkeleynlp/prasann/bin/ops` (stdlib, login-node safe, a few seconds; toolkit at
`/accounts/projects/berkeleynlp/prasann/ops`, docs in `ops/README.md`):

- **Register every launch** (Beaker, slurm, Lambda, local): `ops register --run-id ... --experiment ...
  --cluster ... --job-id ... --expected-minutes N --git <repo>`. Skill: `launch-run`.
- **Job state:** `ops status` (all clusters, one call) and `ops stall <job|run>` for anything flagged.
  No sleep/poll loops — use the Monitor tool for a specific condition. Skill: `job-status`.
- **Numbers:** `ops result add --check` before any number is quoted; investigate every warning.
  **results-hub is retired** (2026-10-12) — nothing is ingested or pushed there. Skill: `record-results`.
- **Experiment state:** the canonical current state of each experiment is its STATUS.md — index at
  `/accounts/projects/berkeleynlp/prasann/ops/experiments/INDEX.md`, one page via `ops exp show NAME`.
  Read it before touching an experiment; update it (skill `close-out-experiment`) before ending a
  session. Memory files and `records/` are history.
- **Hooks:** PreToolUse hooks (`ops/hooks/`) deny foreground poll loops, test runs on the login node,
  NFS interpreters and `/net` paths in jobs, bare `torchrun`, and subagents without a sonnet/opus
  model. A denial is a rule, not a glitch — fix the command.

**CTC evals:** the current path is olmo-eval `scripts/ctc_suite/launch_ctc_suite.py` (tokenizer
`Qwen/Qwen3.5-0.8B`; `--doc-markers` for marker-trained checkpoints; r256k = YaRN x2 + 131,072-token
chunked prefill **by default**, else left-truncated) — see `records/ctc-fast-suite-eval.md` and the
`run-evals` skill. **Canonical olmo-eval branch = `prasann/ctc-suite-flopdrop`** (current `main`, which has PR #400,
+ the launcher and fixes; not merged into main by choice). The old `prasann/ctc-suite*` branches are
superseded. OLMo-core's legacy
native launcher (`recipes/eval/legacy_native/`) remains only for the singletask/xlong ladders.

## Repo layout

| Path | What |
|---|---|
| `src/olmo_core/` (+ `src/test/`) | The library: upstream OLMo-core plus our attention / routing / data extensions. Upstream scripts stay in `src/scripts/` and `src/examples/`. |
| `ctc/` | Vendored CTC task package (generators, seed pools, prompts, graders). `pip install -e ctc/`; provenance in `ctc/VENDORED.md`. Change it upstream and re-vendor. |
| `recipes/` | The canonical, reproducible pipeline — start at `recipes/README.md`. `data/` (setA builder, converter, marker repair), `train/setA/` (dense Qwen3.5-4B setA SFT), `train/research_trainer/` (the task-agnostic torchrun trainer research arms drive), `eval/` (setA audit, legacy native eval, YaRN copies), `env/local_env.sh`. |
| `research/<family>/` | Experiment code by family: `token_routing/`, `ffn_compute/`, `kv_route/`, `landmark/`, plus `common/` (shared helpers, incl. the vendored `corpus_reasoning` package — put `research/common` on the path). Each folder's README names its record. |
| `records/` | Standalone writeups (diagnoses, plans, handoffs, how-tos); index in `records/README.md`. |
| `ops/` | Research-ops toolkit: run registry, job status, results log, `experiments/<name>/STATUS.md` (see **Research ops**). |
| `.claude/skills/` | `run-evals`, `pull-evals`, `storage-audit`, `where-to-run`, `training-smoke-test`, plus `launch-run`, `job-status`, `record-results`, `close-out-experiment` (symlinks into `ops/skills/`, their single source). |

Where new things go:

- **New experiment code goes in `research/<family>/`** — the matching family, or a new family folder
  with a README. Never in a `debug/` directory, and never loose in `src/scripts/` (upstream-only).
- Results, run dirs and weights stay out of git (`research/**/results*/`, `runs/`, `weights/` are
  gitignored): keep them on node-local `/data` or weka and say where in the folder's README.
- Changes to the canonical data → train → eval path go in `recipes/`, and must keep it working end to end.
- **Every experiment also gets one claude.ai artifact** (default since 2026-10-20; global rule in
  `~/.claude/CLAUDE.md`): a summary page kept current by republishing to the same URL, linked from
  STATUS.md `artifacts:` and each record's `artifact:` header. The `close-out-experiment` and
  `record-results` skills include the step.
- New writeups go in `records/`. Every new record starts with the front-matter header documented in
  `records/README.md` (its `summary:` becomes the index row); regenerate the index with
  `ops exp records --write`, never by hand. Current experiment *state* goes in the experiment's
  STATUS.md, not in a record. The repo root keeps only
  README/CHANGELOG/CONTRIBUTING/CLAUDE.md/local_cluster.md/lambda_cluster.md/beaker.md.
- **Data-generation pipeline** (task JSONL → tokenized SFT shards → weka/local staging) lives
  in `recipes/data/` (see `recipes/README.md`): task rows come from the vendored `ctc/` package
  (`ctc-data`) plus two in-repo generators (`recipes/data/generators/`), `ctc_sft/build_ctc_sft.py`
  drives the build, and `convert_unified_to_document_landmark.py` tokenizes to shards.

## Document-chunked / landmark attention: REPAIR THE BASE CHECKPOINT FIRST

Qwen3 never trains the embedding rows for the reserved marker tokens that the document-chunked and
landmark data paths are built on (`<|box_start|>`/`<|box_end|>`, plus the landmark/pad ids past the
real vocab). Their embeddings are **bit-identical** (cosine similarity 1.0000), so the model cannot
distinguish an "open document" marker from a "close document" one. Marker-dense runs then train to
chance and the failure looks exactly like a modeling result.

**Before any document-chunked / landmark training from a fresh base, run
`recipes/data/fix_marker_embeddings.py` on the checkpoint and train from the repaired copy.** The
tokenized shards are correct — do NOT rebuild data. See `records/document-chunked-marker-embeddings.md` for
the full diagnosis, the one-line check for whether a base is affected, and the validation numbers.

⚠ **Fixing the marker cosine is NOT enough — the marker NORM matters too.** The first version of
`fix_marker_embeddings.py` made the markers mutually distinguishable but left them at ~1/3.6 the norm
of a real token, which RMSNorm amplifies into full-strength noise. On the leak-free (label-inside-chunk)
shards this flatlines training at CE ≈ 0.79 for **every** mask, including plain causal — an
unrestricted model cannot even memorize the data, so it reads as "the mask is too restrictive" when it
is not. The script now seeds each marker from a real trained delimiter row (`«`/`»`/…) and asserts the
norm is in-distribution. Any base repaired before 2026-07-14 is affected: re-run the script.
See `records/n100-chunked-marker-position-bug.md`.

## Local (Berkeley) cluster

We also train/eval on Berkeley slurm H200 nodes without AI2 infra (no weka, no Beaker). The full
pipeline — nodes/QOS, the NFS-vs-`/data` rule, env setup, the weka-free training recipe, and the
known traps — is documented in `local_cluster.md`. Read it before touching any `*local*.sbatch`
launcher or debugging a hung/crashed local run.

**berkeleynlp partition = small-scale debugging only, `--qos=normal` only** (2026-10-16; global
rule in `~/.claude/CLAUDE.md`, hook-enforced). jsteinhardt and Lambda are unaffected.

**`/scratch` IS NFS — it is not a fast local disk.** Both `/accounts` (home, this repo) and
`/scratch` are shared NFS: `/scratch` writes at roughly 5 MB/s, and importing from an NFS conda env
costs a ~5–60 s tax per process. The ONLY node-local fast storage is the target node's `/data`
(and `/tmp`). So moving job I/O from `/accounts` to `/scratch` is **not** a mitigation for an NFS
problem — it is the same class of storage, and it will still deadlock concurrent readers/writers in
`nfs_wait_bit_killable`. Job logs, work dirs, JIT/index caches (triton, flashinfer, pyserini) and
checkpoints all belong on `/data`; `/scratch` is fine for code and small artifacts you read once.

**`/net/<node>/...` is for INSPECTION ONLY.** Every compute node's disk is exported at
`/net/<hostname>` (e.g. `/net/horton/data/prasann/...`; nodes: `balrog`, `cubbins`, `feanor`, `horton`,
`lorax`, `mcfuzz`, `mooney`, `rainbowquartz`, `saruman`, `shadowfax`, `smaug`, `smokyquartz`,
`sneetches`, `sunstone`, `thidwick`) — handy for checking what data or checkpoints exist, reading a job
log or confirming a schema without an `srun`; keep it light. But `/net` *is* the NFS layer: pointing
job I/O, an interpreter, a checkpoint read or a cache directory at it recreates the
`nfs_wait_bit_killable` deadlock. A job on horton uses `/data/...`, never `/net/horton/data/...`.
Recursive `find` across `/net` hits command timeouts; go straight to the directory you want.

**The INTERPRETER itself must be node-local for any torch/vLLM job.** The "~5–60 s import tax"
above is the *best* case (small scripts). A vLLM or torch job launched with an NFS interpreter (e.g.
the former `/scratch/.../corpus-reasoning-eval/bin/python`) has to page multiple GB of
shared objects over a ~5 MB/s link and parks in `D` state / `nfs_wait_bit_killable` at ~0% CPU for
many minutes — it looks like a hung GPU or a slow model load, and it is neither. Use the
node-local venv **`/data/prasann/ctc_vllm_venv/bin/python`** (vllm 0.25.1 / torch 2.11.0+cu130)
for vLLM work, together with `export CUDA_HOME=/usr/local/cuda-12.8` and
`export PATH=$CUDA_HOME/bin:$PATH` for the GDN JIT. Diagnose with
`ps -o stat,wchan,%cpu` — `Dl` + `nfs_wait_bit_killabl` + ~0% CPU is this bug, not a compile.
The old NFS env (`/scratch/users/prasann/conda/envs/corpus-reasoning-eval`) no longer exists (removed
2026-09-23); its node-local copy is `/data/prasann/conda/envs/corpus-reasoning-eval` on horton, but for
vLLM use `/data/prasann/ctc_vllm_venv/bin/python` as above. For training/tests use
`/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python`, launched as `$PY -m torch.distributed.run`
(bare `torchrun` resolves to the NFS system miniforge).

**Checking on local jobs:** `ops status` (one cached snapshot of slurm + Beaker + Lambda — never poll
`squeue`/`beaker` in loops), `ops stall <job|run>` (verdict first: NFS stall, GPU idle, hung, no
output for N min), and `ops guard -- CMD` for long local commands outside slurm. Register sbatch
launches with `ops register`, or run the payload through the shared wrapper
`recipes/env/ops_job.sh` (registers, runs under `ops guard`, records the final state; header has the
knobs; see `ops/README.md`).

### Loading an olmo-exported Qwen3.5 in vLLM: use the serving copy

`export_olmo_to_hf.py` emits a **text-only** checkpoint (`model_type: qwen3_5_text`, flat config,
no `vision_config`). vLLM 0.25.1 resolves *any* `Qwen3_5*` architecture to a **multimodal** class
whose `__init__` reads `config.vision_config`, so pointing vLLM at the raw export — even with
`hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]}` — dies at model construction with
`AttributeError: 'Qwen3_5TextConfig' object has no attribute 'vision_config'`, before any memory
is touched.

**FIRST, look for an existing serving copy — do not rebuild one.** Prebuilt, already-validated
copies live at `/data/prasann/ctc_suite/vllm_serving_4b{,_v2,_v3}/<ckpt-name>/` on horton (`_v3`
is current). Point vLLM straight at that directory.

Building one from scratch takes **three** scripts in `research/landmark/ctc_vllm_validation/`, not one — miss
any and the load dies:
1. `make_vllm_serving_copy.py` — wrapper `config.json` (base VL config with our `text_config`).
2. `make_vl_weights.py` — key rename `model.*` → `model.language_model.*`. Without it:
   `ValueError: There is no module or parameter named 'model' in Qwen3_5ForConditionalGeneration`.
3. `add_dummy_visual.py` — ~297 synthesized `visual.*` params. Without them vLLM's
   `track_weights_loading` errors on the uninitialized vision tower.

Sanity-check a serving dir before launching: it should have **~426 `model.language_model.*` and
~297 `visual.*`** keys, and `vision_config` in `config.json`.

The arch override and `limit_mm_per_prompt={"image": 0, "video": 0}` are *part* of the recipe, not
the whole of it — see [[qwen35-4b-vllm-load-recipe]] for all seven pieces. Copying two lines out of
`run_vllm_eval.py` without the serving copy is a repeat failure; that driver is always pointed at
a serving copy someone else already built.

**`hf_overrides` REPLACES a nested sub-config, it does not merge.** Passing
`hf_overrides={"text_config": {"rope_scaling": ...}}` wipes every other `text_config` field and
fails pydantic validation with *"text_config … does not have `num_attention_heads`"*. To change
RoPE/YaRN or any other nested field, patch `config.json` on disk in a symlinked copy (see
`recipes/eval/make_yarn_copy.py`) rather than fighting override semantics.

## Lambda cluster

A third pool: a **separate A100 SLURM cluster reached via `ssh lambda`**, sharing no filesystem, no
git remote, and no internet with the Berkeley cluster. Use it to drain a queue of independent
training runs when the Berkeley per-user GPU quota is saturated — its GPUs are often idle while
jsteinhardt is contended.

**Fair-share rule (Lambda only): cap your *preempting* footprint at two nodes — at most 8 GPUs on
`preemptive_high` and 8 on `preemptive` at any time. Take anything beyond that at `normal`, which
can't preempt and is itself preemptible.** The cluster is shared with one other regular user, and
this is what guarantees both of you can always get 2 nodes. Do NOT carry over the Berkeley
"default to highest QOS and preempt freely" directive.

**`lambda_cluster.md` (repo root)** documents access, the QOS table and that policy, the per-user
1.30T `/accounts` quota (the usual blocker) vs. node-local disk, the rsync staging recipe,
`run_ctc_lambda.sbatch` (`recipes/train/research_trainer/`) knobs, and the trap index. Read it
before touching any `*lambda*.sbatch` launcher. Two traps worth knowing up front: the launcher
header hardcodes `--gres=gpu:A100:8` so a direct execution eats your whole GPU quota, and **`sbatch` routinely
reports `FAILED` on runs that fully succeeded** — verify via the loss curve and checkpoint, never
the exit code.

## Beaker launches

**`beaker.md` (repo root)** is the practical how-to: launch/gantry templates, monitoring/cancel
commands that actually work, the weka-vs-S3 staging two-step, and the traps index. Image and gantry
mechanics: `docs/claude/upstream-olmo-core.md`. Gantry runs the **pushed** commit. Register every
job (`ops register --cluster beaker --job-id <id>`) and check it with `ops status` / `ops stall <id>`.

**Job priority — ALWAYS `urgent`, never lower.** Every Beaker/gantry job (training, eval, data build) must launch at `priority="urgent"` — exactly `urgent`: never `normal`/`high`/`low`, and not `immediate` either. Lower-priority jobs pend behind capacity; `urgent` preempts to get nodes. Set `BeakerLaunchConfig.priority = "urgent"`, pass `--priority urgent` to launchers, or bump a live job with `beaker job update-priority <job-id> urgent`. If you add or edit a launcher, make `urgent` its default.

**minRuntime — ALWAYS set it (default `1h`).** Urgent jobs with `minRuntime=0` get a very long wait
tail: over 14 days (to 2026-10-20) our 0s jobs had p90 waits of 10-12 h (amandab's 117 h), and every
job stuck for days was a 0s job; with a minRuntime, p90 was ~1-1.5 h. Avoid 4-GPU requests on jupiter
when 2 or 8 will do (4 was the slowest size we hit).
`beaker_ctc_suite.py --min-runtime` and the legacy multirung eval launcher (`BEAKER_MIN_RUNTIME`) default to
1h; for gantry pass `--min-runtime 1h`, for `BeakerLaunchConfig` set `min_runtime="1h"`. A job stuck
pending at `0s` must be cancelled and relaunched. Memory: [[beaker-min-runtime]].

**Allocation — ALWAYS unallocated.** Beaker capacity is split into per-budget *allocations*
(`beaker allocation list ai2/jupiter-cirrascale-2 --budget ai2/oe-other`: 14%/168 slots on
jupiter, 15%/88 on ceres, all of it granted to the `ai2/oe-scaling` workspace group; nothing on
saturn/neptune). A job is *allocated* only when its workspace is in that group, and allocated
jobs consume the team's guaranteed quota. Our jobs must stay **unallocated + urgent**: launch
from workspace `ai2/flex2` under budget `ai2/oe-other` and never from an oe-scaling workspace.
There is no per-job flag; the workspace decides. Verify with
`beaker report gpu-usage --organizations ai2 --users prasanns --since 7d -g allocated,cluster`
(non-admins must pass the org filter). Status 2026-09-02: 100% of our GPU hours were unallocated.

## Testing

Conventions are in `docs/claude/upstream-olmo-core.md` (`src/test/` mirrors the source; GPU tests use
`@pytest.mark.gpu`). Run pytest / `make checks` on a compute node with the node-local env, never on the
login node; `ops/tests` (stdlib, < 10 s) is the exception.

## Reporting experimental results

**`n` means CORPUS SIZE — never eval-set size.** In this project `n` is reserved for the number of
documents/claims in an example's context (`n=20`, `n=100`, the `..._n100_k3` files, `--ndocs`). It is
a property of the *task*. Using the same letter for how many examples an eval ran on has already
caused real confusion, so:

- For eval-set size write **`eval_size`** (or spell it out: "488 eval examples"). Never `n=488`.
- New eval scripts must emit the field `eval_size` in their result JSON. Older results wrote this as
  `n`, which is exactly the collision being retired — when reading them, translate to `eval_size`
  rather than propagating the old name.
- The same applies in prose, tables, run names, and anything recorded with `ops result add`.

**Eval sets should have at least 500 examples, and a smaller one must be flagged.** If a number came
from fewer than 500 eval examples, say so *inline, next to the number* — give the size and its error
bar, e.g. `f1=0.83  ⚠ eval_size=100 only (±0.038)`. Never present a sub-500 result bare; a small eval
inflates noise into apparent findings. (The contradiction held-out set is 488 examples, which is the
entire file and is accepted as-is.)

**Quote a resolution, not three decimals.** On a right/wrong-graded eval, the binomial standard error
is ±0.021 at f1≈0.70 and ±0.010 at f1≈0.95 for 488 examples. Before calling a difference real, check
it against that — and remember eval noise is only half the story, since run-to-run seed variation adds
more.
