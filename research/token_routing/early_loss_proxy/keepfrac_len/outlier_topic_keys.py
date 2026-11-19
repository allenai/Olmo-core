"""Outlier topic map, step 1 (node holding the pool): decode every outlier passage of a tokenized pool and hash its
first 100 whitespace-normalized characters (sha1[:16]). Used when the generator jsonl is gone (kl16k_x3p2's x16build
rows). Edit P for another pool. Output (stdout): [{fp, gold, from_pool, src_idx, keys}] for outlier_topic_map.py.
"""
import json, re, hashlib, sys, numpy as np
from transformers import AutoTokenizer
P = "/data/prasann/early_loss_proxy/shards/kl16k_x3p2"
tok = AutoTokenizer.from_pretrained("/data/prasann/hf_models/Qwen3.5-0.8B-Base")
rows = sorted([r for r in map(json.loads, open(f"{P}/row_info.jsonl")) if r["part"] == 1], key=lambda r: r["row"])
ids = np.fromfile(f"{P}/token_ids_part_t1_000.npy", dtype=np.uint32)
off = np.concatenate([[0], np.cumsum([r["len"] for r in rows])])
pat = re.compile(r"^\s*<\|box_start\|>\s*Document \[\d+\]:\s*", re.S)
out, nbad = [], 0
for k, r in enumerate(rows):
    x = ids[off[k]:off[k + 1]].tolist()
    keys = []
    for s, e in r["doc_spans"]:
        t = tok.decode(x[s:e])
        m = pat.match(t)
        if not m: nbad += 1
        t = t[m.end():] if m else t
        t = t.replace("<|box_end|>", "")
        keys.append(hashlib.sha1(" ".join(t.split())[:100].encode()).hexdigest()[:16])
    out.append({"fp": r["fingerprint"], "gold": r["gold"], "from_pool": r.get("from_pool"), "src_idx": r["src_idx"], "keys": keys})
print("rows", len(out), "unparsed docs", nbad, file=sys.stderr)
print(json.dumps(out))
