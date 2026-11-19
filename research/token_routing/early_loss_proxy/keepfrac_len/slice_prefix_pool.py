"""First N rows (row order) of each part of kl16k_x3p2 -> kl16k_x3p2_n<N> (N=3440: 645 steps x 16 rows, so the LR schedule
covers exactly the run). Sidecars filtered. Run on sneetches."""
# first N rows (row order) of each part of kl16k_x3p2 -> kl16k_x3p2_n<N>; same sidecars, filtered
import json, os, collections, numpy as np
N = 3440
src = "/data/prasann/early_loss_proxy/shards/kl16k_x3p2"; out = f"{src}_n{N}"; tmp = out + "_tmp"
os.makedirs(tmp, exist_ok=True)
meta = json.load(open(f"{src}/metadata.json"))
info = [json.loads(x) for x in open(f"{src}/row_info.jsonl")]
gold = json.load(open(f"{src}/gold_fingerprints.json")); pairs = json.load(open(f"{src}/gold_pairs_fingerprints.json"))
new, g2, p2, contra, tot, loss, lens = [], {}, {}, [], 0, 0, []
for pt in sorted({x["part"] for x in info}):
    rows = sorted([x for x in info if x["part"] == pt], key=lambda x: x["row"])
    ids = np.fromfile(f"{src}/token_ids_part_t{pt}_000.npy", dtype=meta["dtype"])
    mk = np.fromfile(f"{src}/labels_mask_t{pt}_000.npy", dtype=meta["mask_dtype"])
    off = np.concatenate([[0], np.cumsum([x["len"] for x in rows])]); assert off[-1] == ids.size
    e = off[N]
    ids[:e].tofile(f"{tmp}/token_ids_part_t{pt}_000.npy"); mk[:e].tofile(f"{tmp}/labels_mask_t{pt}_000.npy")
    tot += int(e); loss += int(mk[:e].sum())
    print(pt, collections.Counter(x.get("from_pool") for x in rows[:N]))
    for x in rows[:N]:
        new.append(x); lens.append(x["len"]); fp = x["fingerprint"]; g2[fp] = gold[fp]
        if fp in pairs: p2[fp] = pairs[fp]
        if x["task"] == "contradiction": contra.append(fp)
with open(f"{tmp}/row_info.jsonl", "w") as f:
    f.writelines(json.dumps(x) + "\n" for x in new)
json.dump(g2, open(f"{tmp}/gold_fingerprints.json", "w")); json.dump(p2, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
json.dump(contra, open(f"{tmp}/gdd_only_contradiction.json", "w"))
meta.update({"task": os.path.basename(out), "num_instances": len(new), "num_tokens": tot, "num_loss_tokens": loss,
             "max_example_len": max(lens), "min_example_len": min(lens), "sliced_from": src, "rows_per_part": N})
json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
os.rename(tmp, out); print("pool", out, len(new), tot)
