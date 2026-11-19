"""Arm E pool: three-task LENGTH MIX of native rows. Per task n16 rows from kl16k_x3p2_n3440 + n2/n4/n8 rows (first in row
order) from kl2k/kl4k/kl8k, shuffled within each task part.
    $PY build_lenmix_pool.py 172 1138 1200 930   # -> shards/kl16k_lenmixE_172_1138_1200_930 (0.521 PF/step measured)
"""
# three-task length-mix pool for arm E: per task n16 rows from kl16k_x3p2_n3440 + n2/n4/n8 rows from kl2k/kl4k/kl8k
import json, os, sys, random, collections, numpy as np
n16, n2, n4, n8 = map(int, sys.argv[1:5])
S = "/data/prasann/early_loss_proxy/shards"
src = {"16k": f"{S}/kl16k_x3p2_n3440", "2k": f"{S}/kl2k", "4k": f"{S}/kl4k", "8k": f"{S}/kl8k"}
want = {"16k": n16, "2k": n2, "4k": n4, "8k": n8}
out = f"{S}/kl16k_lenmixE_{n16}_{n2}_{n4}_{n8}"; tmp = out + "_tmp"; os.makedirs(tmp, exist_ok=True)
meta0 = json.load(open(f"{src['16k']}/metadata.json"))
data = {}
for L, p in src.items():
    m = json.load(open(f"{p}/metadata.json")); info = [json.loads(x) for x in open(f"{p}/row_info.jsonl")]
    assert m["eos_token_id"] == meta0["eos_token_id"] and m["doc_start_id"] == meta0["doc_start_id"] and m["tokenizer"] == meta0["tokenizer"]
    assert m["query_position"] == meta0["query_position"] and m["wrap_docs"] == meta0["wrap_docs"]
    g = json.load(open(f"{p}/gold_fingerprints.json"))
    for pt in sorted({x["part"] for x in info}):
        rows = sorted([x for x in info if x["part"] == pt], key=lambda x: x["row"])
        ids = np.fromfile(f"{p}/token_ids_part_t{pt}_000.npy", dtype=m["dtype"]); mk = np.fromfile(f"{p}/labels_mask_t{pt}_000.npy", dtype=m["mask_dtype"])
        off = np.concatenate([[0], np.cumsum([x["len"] for x in rows])]); assert off[-1] == ids.size == mk.size, (L, pt)
        task = rows[0]["task"]
        assert len(rows) >= want[L], (L, task, len(rows))
        for k in range(want[L]):
            data.setdefault(task, []).append((L, rows[k], ids[off[k]:off[k + 1]], mk[off[k]:off[k + 1]], g.get(rows[k]["fingerprint"])))
rng = random.Random(20261119)
new, gold, stats, tot, loss, lens = [], {}, collections.defaultdict(list), 0, 0, []
for pt, task in enumerate(["contradiction", "outlier", "qdmatch_nq"]):
    rs = data[task]; rng.shuffle(rs)
    np.concatenate([r[2] for r in rs]).astype(meta0["dtype"]).tofile(f"{tmp}/token_ids_part_t{pt}_000.npy")
    np.concatenate([r[3] for r in rs]).astype(meta0["mask_dtype"]).tofile(f"{tmp}/labels_mask_t{pt}_000.npy")
    for k, (L, x, a, b, gg) in enumerate(rs):
        y = dict(x, part=pt, row=k, len_pool=L); new.append(y); lens.append(len(a)); tot += len(a); loss += int(b.sum())
        stats[(task, L)].append(len(a))
        if gg is not None: gold[x["fingerprint"]] = gg
with open(f"{tmp}/row_info.jsonl", "w") as f: f.writelines(json.dumps(x) + "\n" for x in new)
json.dump(gold, open(f"{tmp}/gold_fingerprints.json", "w")); json.dump({}, open(f"{tmp}/gold_pairs_fingerprints.json", "w"))
meta = dict(meta0); meta.update({"task": os.path.basename(out), "num_instances": len(new), "num_tokens": tot, "num_loss_tokens": loss,
    "max_example_len": max(lens), "min_example_len": min(lens), "lenmix": {"16k": n16, "2k": n2, "4k": n4, "8k": n8, "sources": src}})
json.dump(meta, open(f"{tmp}/metadata.json", "w"), indent=1)
os.rename(tmp, out)
print("pool", out, len(new), "mean tokens/row", tot / len(new))
for k in sorted(stats): print(k, len(stats[k]), int(np.mean(stats[k])))
