# early_loss_proxy: does the dense eval loss after a few training steps predict a FLOP-drop method's final score?

Record: [`records/early-dense-loss-proxy.md`](../../../records/early-dense-loss-proxy.md) (experiment `tokdrop_lr`).

Each arm trains with its own method (routed / dropped forward, optimiser, LR and the real schedule of its full
488-step run: linear warmup over the first 3% = 15 steps, then linear decay) from the repaired Qwen3.5-4B base on the
first batches of the 4-task pool `mt4f1952` (same order for every arm at a given seed), and at step 0 and after
each probe step it scores the DENSE forward (model.eval(): full context, no dropping, no router, LoRA adapters on)
on fixed held-out rows: teacher-forced answer CE, per row = mean over the answer tokens.

| file | what |
|---|---|
| `build_heldout.py` | held-out rows (CPU, on the node): `suite8k` / `suite2k` = the first rows of the 4 eval rung files the final score uses (tokenized like the training shards); `train` = rows 1952.. of `mt4f3904` (never trained on; f3904 is a byte prefix extension of f1952, checked). Prompt-hash overlap with the 7,808 training rows: 0 |
| `dense_probe.py` | the probe: `DenseProbeCallback` attached to the unmodified `train_ctc_suite.py`; hard-stops at the last probe step, saves nothing; per-step wall clock, LR, FLOPs (flop meter) and batch hash |
| `launch_proxy.py`, `run_proxy.sbatch`, `stage_code.sh` | arms (flag for flag the arms' full-run commands) -> one slurm job running the arm runs back to back on a jsteinhardt node (2 GPUs); registered with `ops` |
| `analyze.py` | curves, drops, fraction of the dense drop (same step / same FLOPs), paired SE vs dense, Spearman vs the known final 8k mean (overall and per task), figure |
| `results_<tag>.json`, `proxy_<tag>.png` | results (raw per-run JSONs + logs stay on `mooney:/data/prasann/early_loss_proxy/results/<tag>/`) |
| `experiment_log_entry.json` | entry for the hub's experiment log |

```bash
bash research/token_routing/early_loss_proxy/stage_code.sh mooney          # -> /data/prasann/early_loss_proxy/code_<sha>
# once per node: $PY research/token_routing/early_loss_proxy/build_heldout.py --out /data/prasann/early_loss_proxy/heldout_v1.npz
python3 research/token_routing/early_loss_proxy/launch_proxy.py --code /data/prasann/early_loss_proxy/code_<sha> \
    --arms recipe --seeds 0 --steps 10 --sets suite8k:16 --tag mycheck
python3 research/token_routing/early_loss_proxy/analyze.py --results /net/mooney/data/prasann/early_loss_proxy/results/v1 \
    --tag v1 --out research/token_routing/early_loss_proxy/results_v1.json --fig research/token_routing/early_loss_proxy/proxy_v1.png
```

Cost: see the record (load, per-step and per-probe wall clock are in every run JSON).

Ported for this check (minimal hunks; FLOP accounting of the tail rule not ported, irrelevant here): oracle token
selection (`src/olmo_core/nn/oracle_token_select.py` + router `score="oracle"`) from `claude/oracle-thresh-sweep-69bc7d`,
and the content-blind tail rule (`--st-otr-score tail`) from `claude/posid-reset-0abac3`.
