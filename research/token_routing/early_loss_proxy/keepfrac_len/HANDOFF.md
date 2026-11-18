# keepfrac_len handoff

## 2026-11-18 15:40 PT (8k gold-drop check, Prasann)
- Asked: does gold dropping work at 8k like at 32k/64k? Scope (revised): `grJ` only (all rows), seeds 0 and 1, kl8k,
  full epoch, seq-len 10,240. Beaker elp-kl8k-bkRPQTGH7X (sd0, 01M7RG89GT036G206PRPQTGH7X) and elp-kl8k-bk3M9X4FBD
  (sd1, 01M7RG963FESHG1R6N3M9X4FBD). grJc was NOT submitted (launcher stopped before its first submit).
- kl8k has gold_fingerprints.json and gdd_only_contradiction.json; gold share per task (row_info gold / n_docs):
  8k 3.2% / 5.3% / 7.1% (contradiction / outlier / qdmatch_nq), 32k 0.9 / 1.6 / 1.9%, 64k 0.5 / 0.8 / 0.9%: all
  < 0.5, so grJ drops documents on all three tasks (incl. qdmatch_nq).
- Then: `[gold-doc-drop]` log counts per arm, score, fold 64k k70 (landed), record / log entry / ops rows.

## 2026-11-17 19:35 PT (takeover session, latest)
- 64k seed 1 (k40/k20/k10) and 32k k70 seed 1 landed, scored (finalize.sh), record verdict + hub head + log entry
  updated, 20 new ops rows added (411 total for this study). Only 64k k70 sd0 is still running
  (elp-kl64k-bkBX8XWK0M = 01M7P6T8EEKK6T00K0BX8XWK0M, ~21:30 PT). When it lands: `bash finalize.sh`, add the 64k k70
  line to the record verdict / hub head / experiment_log_entry.json (then status done), and add its ops rows
  (run id elp-kl64k-k70-sd0-probe only).

## 2026-11-17 18:25 PT
- Record written: `records/keep-fraction-vs-length.md` (verdict: best keep 20-40% at every resolvable length, does
  not fall; low keeps break harder with length; constant fraction fits). Chart `keepfrac_vs_length.png`, hub block
  `hub_keeplen_block.html`, `experiment_log_entry.json` (2 entries: tracks 64k [running] and mechanism [done]),
  391 rows in ops results (`record_results.py`).
- STILL RUNNING: 64k seed 1 of k40/k20/k10 (elp-kl64k-bkWA90AAKH, -bkB456AQ25, -bkP1WMGF1Y, ~19:00), 32k k70 sd1
  (elp-kl32k-bkR20896XN, ~19:30), 64k k70 sd0 (elp-kl64k-bkBX8XWK0M, ~21:30).
- When they land: `bash finalize.sh` (rescore, refresh tables/chart/block), update the verdict lines in the record,
  `experiment_log_entry.json` (status done) and the hub head; add only the NEW rows to ops (filter
  `record_results.py` output to the new run ids: elp-kl64k-{k40,k20,k10}-sd1-probe, elp-kl32k-k70-sd1-probe,
  elp-kl64k-k70-sd0-probe).

## 2026-11-17 15:25 PT (takeover session)
- q32k scored: `INTERIM_q32k.md`, `q32k_scored.json` (k40 ~ k20 best; k10 and below plateau higher).
- launch_kl.py: `--local CODE_DIR` (jsteinhardt via launch_proxy.py, node /data paths, MICRO env, `--group` arms/job),
  `--heldout` (default heldout_kls.npz at 2k/4k/8k), lengths 2k/4k/8k added.
- q64k quick read RUNNING (Prasann asked 15:15): pool kl64k, held-out heldout_kl.npz set kl64k:48 (eval_size 144),
  steps 10-40, seed 0, micro 1, code_7c6b5d877, mooney preemptive_high, 4 x 2 GPUs: slurm 5201918 (dft,k5),
  5201919 (ref,k10), 5201920 (k40,k2p5), 5201921 (k20,k1p25). Results mooney results/q64k/.
  Score: copy to <dir>/q64k, `analyze_kl.py --root <dir> --lens 64k --tag-prefix q --ref dft`.
- FULL WAVES (seed 0, 225 steps = one epoch, probes 10,25,...,225, held-out 48 rows/task = eval_size 144, Beaker
  2 x H100 unless noted, results weka/S3 `early_loss_proxy/results/kl<L>/`, fetch `../fetch_beaker_results.sh kl<L>`;
  ops run ids `elp-kl<L>-bk<last 8 of exp id>`, see `ops runs --experiment tokdrop_lr`):
  - 32k: dft (relaunched micro 1 after an OOM at micro 2: elp-kl32k-bk7Q0HNJB8), ref, k40, k20, grJ, grJc, k10, k5, k2p5.
  - 16k: dft, ref, k70, k40, k20, k10, k5, k2p5.
  - 64k: k40, k20, k10, k5, k2p5, k1p25, grJ, grJc (2 GPUs, micro 1); dft + ref on 8 GPUs (2 GPUs OOM / > 4 h:
    dft at 64k reserves 78 GiB on an H200 at 76 s/step): elp-kl64k-bkW1T679MP, elp-kl64k-bkZZEQ26K6.
  - 2k/4k/8k: first launch failed (trainer refuses rows > --seq-len; the short pools overshoot their label, max row
    3,367 / 5,354 / 9,819), marked failed in ops; relaunched with seq-len 4096 / 6144 / 10240 (wave 4), arms dft,
    ref, k70, k40, k20, k10, k5, k2p5.
- Score: fetch each tag into one dir, `analyze_kl.py --root <dir> --lens 2k,...,64k --ref dft --out kl_results.json`,
  then `summarize_kl.py --scored kl_results.json --out-json kl_summary.json --png keepfrac_vs_length.png`.
- 16:40 wave 5: 32k k70 sd0 (k40 beat k20 1.27x at 32k) + seed 1 of 32k k40/k20/k10/grJc, 64k k40/k20/k10,
  16k k40/k20/k10, 8k k70/k40/k20 (ops ids in `ops runs`, tags kl<L>).
- Scored so far (score_all.sh -> /tmp/claude-3018/keepfrac_len_results/summary.txt): 2k, 4k, 8k, 16k complete for
  seed 0; 32k router arms done, 32k dft still running (micro-1 relaunch); 64k all running (dft/ref on 8 GPUs).
  Early read: best keep ~40% at 8k/16k/32k, not lower; 2k/4k saturate (held-out CE ~0.01: unresolvable).
- Next: when 32k dft + 64k land, rerun score_all.sh, write records/keep-fraction-vs-length.md, chart, ops result
  add --check, experiment_log_entry.json, hub_keeplen_block.html.

## previous session (2026-11-17 ~15:10 PT, usage limit)

Done:
- Pools built on mooney `/data/prasann/early_loss_proxy/shards/kl{2k,4k,8k,16k,32k,64k}` (3 tasks x 1,200 rows,
  gold checks pass, 0 prompt hits) + `heldout_kl.npz` (kl16k/32k/64k) and `heldout_kls.npz` (kl2k/4k/8k), 48 rows/task.
- Staging to weka started (stage_to_weka.sh mooney heldout_kl.npz heldout_kls.npz shards/kl*); check the Beaker
  `elp-stage-weka` job / `ls /weka/.../early_loss_proxy/shards/kl64k` before launching on Beaker.
- Quick 32k read (tag q32k, mooney results/q32k/, pool mtb64k_r32k = mtb64k's 960 32k rows, held-out heldout_v4
  gen32k, eval_size 72, 40 steps, one seed): ref (dense LoRA), dft (dense full FT, primary reference), k40/k20/k10/k5/k2p5.
  dft and k2p5 were still running at handoff. Score: copy results/q32k to a folder, then
  `python3 analyze_kl.py --root <dir-with-q32k> --lens 32k --tag-prefix q --ref dft`.

Next:
1. Score q32k, send interim note to the orchestrator (local_a8068d93-...).
2. Beaker waves via launch_kl.py (dft, ref, k40..k2p5, +k70 at short lengths if k40 > k20, +k1p25 at 64k,
   c1k10/c2k3p5, grJ/grJc at 32k/64k), full epoch = 225 steps; 2 seeds on deciding arms. Short lengths use
   `--heldout heldout_kls.npz` (launch_kl.py currently hardcodes heldout_kl.npz: add the switch).
3. Record, chart, ops result add --check, experiment_log_entry.json, hub keeplen block.
