# attn_groups: group-partitioned attention on top of gold-document drop (256k contradiction)

Prasann (2026-11-20). Library: `src/olmo_core/nn/attention/group_attention.py` (tests
`src/test/nn/attention/group_attention_test.py`); trainer flags `--st-attn-groups G --st-attn-group-decoys K
--st-attn-group-full-frac p` (online-router soft-token variant only; default off = exact no-op). Record: section
"Group-partitioned attention" in `records/longctx-contradiction-golddrop.md`.

In the attention layers only, each training row's documents are split every forward into hubs (gold + K random decoy
documents + preamble) and G random groups of the rest; group tokens attend to their group + hubs, hub tokens to hubs,
the question/answer strip to everything; a share p of rows keeps full attention. Each set runs as its own gathered
flash-attn varlen sequence (a real compute saving), in the chunked prefix (`--prefix-chunk`) and after the router's
mid-stack drops too. Scoring is unchanged (eval = full attention).

| file | what |
|---|---|
| `screen_local.sh` | per-step PF / s / peak GiB screen on 16 real lc256kmix_gdd 256k rows, 1 GPU per config (jsteinhardt) |
| `screen_table.py` | screen table (paired per step vs attention off) |
| `compare_ag.py` | the arm vs gdd256 / dense256b: matched-PF, FLOPs-to-match, matched-rows, per-step cost |
| `../launch_arms.sh gddag` | the Beaker training arm (`G= SI= NAME= FF= MAXPF=`); score with `launch_arms.sh score <name> <steps>` |

Screen outputs: cubbins `/data/prasann/attn_groups/s1/out{1,2}/` (drop / whole rows). Arm checkpoints: weka
`.../early_loss_proxy/runs_lc/gddag4/stepN`; results S3 `early_loss_proxy/results/lcA-gddag4`, `lcS-gddag4`.
