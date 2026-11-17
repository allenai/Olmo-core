# keepfrac_len handoff

## 2026-11-17 15:45 PT (takeover session)
- q32k scored: `INTERIM_q32k.md`, `q32k_scored.json` (k40 ~ k20 best; k10 and below plateau higher).
- launch_kl.py: `--local CODE_DIR` (jsteinhardt via launch_proxy.py, node /data paths, MICRO env, `--group` arms/job),
  `--heldout` (default heldout_kls.npz at 2k/4k/8k), lengths 2k/4k/8k added.
- q64k quick read RUNNING (Prasann asked 15:30): pool kl64k, held-out heldout_kl.npz set kl64k:48 (eval_size 144),
  steps 10-40, seed 0, micro 1, code_7c6b5d877, mooney preemptive_high, 4 x 2 GPUs: slurm 5201918 (dft,k5),
  5201919 (ref,k10), 5201920 (k40,k2p5), 5201921 (k20,k1p25). Results mooney results/q64k/.
  Score: copy to <dir>/q64k, `analyze_kl.py --root <dir> --lens 64k --tag-prefix q --ref dft`.

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
