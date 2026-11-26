"""Held-out contradiction rows + gold document indices for the gold-drop gradient probe (tokgrad_probe.py).

16k: heldout_lc.npz set lc16k (ctc eval split, eval seed 99; gold disjoint from gdd256 / dense256b training);
64k: heldout_kl.npz set kl64k/contradiction (eval split; the lc pools dropped any row sharing gold with it).
Gold = P.gold_info on the source jsonl row (src_idx), checked against the token row's document count.

    python3 prep_rows.py --n16 16 --n64 8 --out /scratch/users/prasann/tokgrad/rows.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "early_loss_proxy"))
import build_pool_v2 as P  # noqa: E402

R = "/net/mooney/data/prasann/early_loss_proxy"  # login-node read of two small jsonl + 24 rows (inspection-size)
SRC = {"lc16k": (f"{R}/heldout_lc.npz", "lc16k", f"{R}/v2build/lcS/hogen/16k/contradiction/eval_16k.jsonl"),
       "kl64k": (f"{R}/heldout_kl.npz", "kl64k", f"{R}/v2build/long/heldout_gen/contradiction_64k/contradiction/eval_64k.jsonl")}
ap = argparse.ArgumentParser()
ap.add_argument("--n16", type=int, default=16)
ap.add_argument("--n64", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ids_l, mask_l, gold_l, name_l = [], [], [], []
for key, n in (("lc16k", a.n16), ("kl64k", a.n64)):
    npz, setname, jl = SRC[key]
    z = np.load(npz)
    sets, tasks, src, offs = z["sets"], z["tasks"], z["src_idx"], z["offsets"]
    idx = [i for i in range(len(sets)) if sets[i] == setname and tasks[i] == "contradiction"][:n]
    ids_all, mask_all = z["ids"], z["mask"]
    exs = [json.loads(l) for l in open(jl)]
    for i in idx:
        ids = ids_all[offs[i]:offs[i + 1]].astype(np.int64)
        ex = exs[int(src[i])]
        g, _ = P.gold_info("contradiction", ex)
        assert len(P.doc_spans(ids)) == len(ex["documents"]), (key, i)
        ids_l.append(ids), mask_l.append(mask_all[offs[i]:offs[i + 1]].astype(bool)), gold_l.append(g), name_l.append(key)
    print(key, len(idx), "rows, mean len", np.mean([len(x) for x, s in zip(ids_l, name_l) if s == key]))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
o = np.concatenate([[0], np.cumsum([len(x) for x in ids_l])])
np.savez(a.out, ids=np.concatenate(ids_l), mask=np.concatenate(mask_l), offsets=o, sets=np.array(name_l),
         gold=np.array(json.dumps(gold_l)))
print("wrote", a.out)
