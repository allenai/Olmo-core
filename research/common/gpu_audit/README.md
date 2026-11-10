# gpu_audit

Read-only audit of GPU usage and agent workflow, 2026-10-20 → 2026-11-10, across Beaker, Berkeley slurm
(jsteinhardt, berkeleynlp) and Lambda. Record: `records/gpu-usage-workflow-audit-2026-11.md`. Ops page:
`ops exp show gpu_audit`. Artifact (private): https://claude.ai/artifact/AxWXjBS1jDQy51MtGszd3d

All scripts are stdlib-only and safe on the login node (each run takes a few seconds; `fetch_beaker.py` is
network-bound, about 10 min).

## Re-run

```bash
cd research/common/gpu_audit
F=JobID,JobName%60,User,Partition,QOS,State,ExitCode,Submit,Eligible,Start,End,Elapsed,ElapsedRaw,AllocTRES%80,NodeList,Reason%40,Comment%80
sacct -u prasann -S 2026-10-13 -E 2026-11-11 -X -P -D --format=$F > raw/sacct_berkeley_prasann.psv
ssh lambda "sacct -u \$USER -S 2026-10-13 -E 2026-11-11 -X -P -D --format=$F" > raw/sacct_lambda_prasann.psv
G=JobID,JobName%40,User,Partition,QOS,State,Submit,Start,End,ElapsedRaw,AllocTRES%80,NodeList
sacct -a -r jsteinhardt,berkeleynlp -S 2026-10-20 -E 2026-11-11 -X -P --format=$G > raw/sacct_berkeley_all.psv
ssh lambda "sacct -a -S 2026-10-20 -E 2026-11-11 -X -P --format=$G" > raw/sacct_lambda_all.psv
# Beaker's own accounting (non-admins need the org filter); per-day loop is in analyze.py's docstring inputs
beaker report gpu-usage --organizations ai2 --users prasanns --start 2026-10-20T00:00:00Z --end 2026-11-11T00:00:00Z -g cluster,allocated,workspace-group --format csv
python3 fetch_beaker.py --since 2026-10-13 --out raw/beaker_jobs.jsonl   # per-attempt Beaker timing
python3 analyze.py        # -> data/*.csv, data/summary.json
python3 sessions.py       # -> data/sessions.csv (Claude transcript idle-time breakdown)
python3 build_page.py     # -> page/gpu-audit.html (the artifact; real usernames, gitignored)
```

The per-day Beaker report (`data/beaker_report/bk_report_daily_cluster_alloc.csv`) is one
`beaker report gpu-usage ... -g cluster,allocated` call per day.

## Files

| path | what |
|---|---|
| `fetch_beaker.py` | `beaker experiment get` in batches for every ID in the registry and the ops Beaker cache; one row per job attempt |
| `analyze.py` | unified attempt table, daily and hourly occupancy, cap headroom, waits, waste, latency, process gaps |
| `sessions.py` | idle gaps in Claude transcripts, labeled by what ended them |
| `build_page.py`, `page_template.html` | artifact page |
| `data/jobs.csv` | one row per GPU attempt in the window (7,619 rows): pool, experiment, kind, GPUs, QOS / priority, minRuntime, state, times, wait, GPU-h |
| `data/summary.json` | every headline number |
| `data/hourly_occupancy.csv`, `daily_*.csv`, `waits.csv`, `duplicates.csv` | chart inputs |
| `data/workflow_incidents.csv` | 77 dated incidents from STATUS pages and records |
| `data/beaker_report/` | Beaker `report gpu-usage` pulls |
| `raw/` (gitignored) | sacct dumps, `beaker_jobs.jsonl`, `user_map.json` (pseudonym → user) |

Other cluster users are pseudonymized (`user1`, …) in everything committed, because this repo's remote is public.
