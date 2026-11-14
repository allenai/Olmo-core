# docdrop: random non-gold document dropping (baseline vs the token-routing recipe)

Record: [`records/docdrop-nongold-baseline.md`](../../../records/docdrop-nongold-baseline.md) (experiment `tokdrop_lr`).

For each training row of the 4-task fast8k mix, keep every gold document and a random share `q` of the non-gold
documents, and remove the rest before block 0 (they cost no FLOPs and no memory). Surviving documents keep their
text, so their printed IDs ("Claim 37", "Document [37]") are the full-context IDs and the answer needs no
relabelling. oolong rows stay whole (the label counts over every item). `q` is set so the run's training FLOPs
match the recipe's (~344 PF at 1,952 rows/task). Evaluation is unchanged: dense, full context.

| file | what |
|---|---|
| `docdrop.py` | selection logic (pure numpy): split a row into documents, parse printed / gold IDs, choose the keep set, cut the row |
| `docdrop_test.py` | CPU unit test: kept set = all gold + exactly `keep_count` uniformly random non-gold, nested in `q`; surviving IDs, prompt and labels unchanged |
| `build_docdrop_shard.py` | node-side builder: new shard + `keep_sidecar.json` + stats + decoded examples; bisects `q` on the FLOP model (validated: dense full-FT model 1,528.7 PF vs the dense run's measured 1,531) |
| `launch_docdrop.py`, `run_docdrop_local.sbatch`, `stage_code.sh` | jsteinhardt launch (code staged to node `/data`; registered with `ops`) |
| `collect_docdrop.py` | results table (scores from `ops result`, PF / peak memory from the node log, FTM vs the dense curve) |
| `experiment_log_entry.json` | entry for the hub's experiment log |

Arms (both: whole-model LoRA r64 scale 32, LR 1e-5, 16 rows/step, 2 GPUs, the dense-LoRA arm's flags):

- `ddc`: compact positions. Trains on the shortened rows of `/data/prasann/docdrop/shards/mt4f1952_dd` (mooney).
- `ddp`: original positions. Trains on the full rows of `mt4f1952`; the soft-token compaction removes the same
  dropped documents (from `keep_sidecar.json`, `--st-n-random 0 --st-drop-slots`) and keeps full-context position ids.

Outputs (not in git): shards `mooney:/data/prasann/docdrop/shards/`, runs `mooney:/data/prasann/docdrop/runs/<run>`,
logs `mooney:/data/prasann/joblogs/dd-<arm>-<job>.out`.

```bash
bash research/token_routing/docdrop/stage_code.sh mooney          # -> /data/prasann/docdrop/code_<sha>
# on the node: $PY research/token_routing/docdrop/build_docdrop_shard.py --src /data/prasann/oracle/shards/mt4f1952 \
#     --out /data/prasann/docdrop/shards/mt4f1952_dd --target-pf 344
python3 research/token_routing/docdrop/launch_docdrop.py --arms ddc --code /data/prasann/docdrop/code_<sha>
```

FLOP meter change (2026-11-13): with `--lora-rank` and per-example accounting, the frozen-base / adapter
adjustment is now priced at each example's own length (`src/olmo_core/train/callbacks/flop_meter.py`), not at
`--seq-len`; at `--seq-len 12288` the old version credited a 1.5k-token row with attention it never ran.
