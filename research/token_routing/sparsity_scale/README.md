# sparsity_scale

Does token-drop + layer-skip headroom grow with model size? A frozen sweep on the dense fs35 SFT
checkpoints (contradiction s56M, oolong s80M) at Qwen3.5 0.8B / 2B / 9B / 27B. Plan:
`records/sparsity-scale-plan.md`; state: `ops/experiments/sparsity_scale/STATUS.md`.

| file | what |
|---|---|
| `marker_check.py` | gate before the sweep: the fs35 dense models trained without `<|box_start|>`/`<|box_end|>`, which the drop harnesses need. Per eval row, answer CE with vs without markers (same converter, `doc_markers` True/False). |
| `launch_beaker.py` | one 1-GPU urgent Beaker job per (size, task); checkpoints and eval rows read from weka, JSON written to `/weka/.../prasanns/sparsity_scale/<probe>/`. `--git-ref` runs a pushed commit other than HEAD. |

Results stay on weka (`sparsity_scale/`), not in git.
