"""Outlier topic map, step 2 (mooney, holds outlier.seed.jsonl.gz): map each passage key to its article title(s);
a key shared by several articles is resolved to the candidate title that the row's other passages use most. Emits
{"groups": {fingerprint: [group id per doc] or null}, "bad": [...]} (null = gold is not exactly one 3-passage group with
every other group >= 4; GoldDocDrop passes such rows through whole). kl16k_x3p2: 3,835 / 3,840 rows mapped; agrees with
outlier_topic_map_text.py on 1,187 / 1,188 kl16k rows.
    srun -w mooney ... $PY -c "$(cat outlier_topic_map.py)" < keys.json > groups.json
"""
import gzip, hashlib, json, sys, collections
rows = json.load(sys.stdin)
want = {k for r in rows for k in r["keys"]}
title, multi = {}, 0
with gzip.open("/data/prasann/ctc_seed_pools/outlier.seed.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        for t, bodies in d.get("articles", []):
            for b in bodies:
                k = hashlib.sha1(" ".join(b.split())[:100].encode()).hexdigest()[:16]
                if k in want:
                    title.setdefault(k, set()).add(t)
        del d
multi = sum(len(v) > 1 for v in title.values())
print("passages", len(want), "mapped", len(title), "multi-title keys", multi, file=sys.stderr)
nres = 0
groups, bad, nomap = {}, [], 0
for r in rows:
    cand = [title.get(k, set()) for k in r["keys"]]
    uni = collections.Counter(next(iter(c)) for c in cand if len(c) == 1)
    ts = []
    for c in cand:
        if not c: ts.append(None)
        elif len(c) == 1: ts.append(next(iter(c)))
        else:
            nres += 1; ts.append(max(sorted(c), key=lambda t: uni.get(t, 0)))
    nomap += sum(t is None for t in ts)
    ids = {}
    g = [ids.setdefault(t if t is not None else f"__u{j}", len(ids)) for j, t in enumerate(ts)]
    cnt = collections.Counter(g); gg = {g[i] for i in r["gold"]}
    ok = len(gg) == 1 and cnt[next(iter(gg))] == 3 and all(c >= 4 for x, c in cnt.items() if x not in gg)
    groups[r["fp"]] = g if ok else None
    if not ok: bad.append({"src_idx": r["src_idx"], "from_pool": r["from_pool"], "counts": sorted(cnt.values())[:4]})
print("ambiguous passages resolved by row context", nres, file=sys.stderr)
print("rows", len(rows), "bad", len(bad), "unmapped docs", nomap, file=sys.stderr)
print(json.dumps({"groups": groups, "bad": bad}))
