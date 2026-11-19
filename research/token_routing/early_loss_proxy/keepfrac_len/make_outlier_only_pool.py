"""Outlier-only pool kl16k_outlier (part t1 of kl16k + sidecars) and held-out heldout_kl16k_outlier.npz (the 48 kl16k
outlier rows of heldout_kl.npz). Run on sneetches."""
# outlier-only pool kl16k_outlier (part t1 of kl16k) + outlier-only held-out npz (kl16k outlier 48 rows of heldout_kl.npz)
import json, os, shutil
import numpy as np
R = "/data/prasann/early_loss_proxy"
src, out = f"{R}/shards/kl16k", f"{R}/shards/kl16k_outlier"
tmp = out + "_tmp"
os.makedirs(tmp, exist_ok=True)
for f in ["token_ids_part_t1_000.npy", "labels_mask_t1_000.npy", "gold_fingerprints.json", "gold_pairs_fingerprints.json"]:
    shutil.copy(f"{src}/{f}", f"{tmp}/{f}")
meta = json.load(open(f"{src}/metadata.json"))
info = [json.loads(x) for x in open(f"{src}/row_info.jsonl")]
rows = sorted([x for x in info if x["part"] == 1], key=lambda x: x["row"])
assert all(x["task"] == "outlier" for x in rows) and len(rows) == 1200
ids = np.fromfile(f"{tmp}/token_ids_part_t1_000.npy", dtype=meta["dtype"])
mk = np.fromfile(f"{tmp}/labels_mask_t1_000.npy", dtype=meta["mask_dtype"])
assert sum(x["len"] for x in rows) == ids.size == mk.size
with open(f"{tmp}/row_info.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
json.dump([], open(f"{tmp}/gdd_only_contradiction.json", "w"))
lens = [x["len"] for x in rows]
meta.update({"task": "kl16k_outlier", "num_instances": len(rows), "num_tokens": int(ids.size),
             "num_loss_tokens": int(mk.sum()), "max_example_len": max(lens), "min_example_len": min(lens),
             "sliced_from": src, "sliced_part": 1, "merged_from": [m for m in meta["merged_from"] if m["task"] == "outlier"],
             "tasks_order": ["outlier"]})
json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
os.rename(tmp, out)
print("pool", out, len(rows), ids.size)
z = np.load(f"{R}/heldout_kl.npz")
o = z["offsets"]
idx = [i for i in range(len(z["sets"])) if z["sets"][i] == "kl16k" and z["tasks"][i] == "outlier"]
segs_i = [z["ids"][o[i]:o[i + 1]] for i in idx]
segs_m = [z["mask"][o[i]:o[i + 1]] for i in idx]
offs = np.concatenate([[0], np.cumsum([len(s) for s in segs_i])])
np.savez(f"{R}/heldout_kl16k_outlier.npz", ids=np.concatenate(segs_i), mask=np.concatenate(segs_m), offsets=offs,
         sets=z["sets"][idx], tasks=z["tasks"][idx], src_idx=z["src_idx"][idx])
print("heldout", len(idx))
