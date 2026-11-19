"""Outlier topic map, v1 (exact passage text): per kl16k outlier row, the article title of every passage, matched by
exact text against the seed pool; emits {"groups": {fingerprint: [group id per document]}, "bad": [...]} on stdout.
Runs on mooney (gen jsonl + seed pool). A row is "bad" unless its 3 gold passages form one group and every other group
has >= 4 passages. Superseded by outlier_topic_keys.py + outlier_topic_map.py (that version resolves passages whose
text occurs in several articles; this one keeps the first title it sees, and all 12 rows it rejects are valid under the newer map).
    $PY outlier_topic_map_text.py rows.json > topics_out.json
"""
import gzip, hashlib, json, sys, collections, time
t0 = time.time()
ROWS = json.load(open(sys.argv[1]))  # [{fingerprint, src_idx, gold, n_docs}] from the pool's row_info.jsonl
SRC = "/data/prasann/early_loss_proxy/v2build/kl/gen/outlier_16k/outlier/train.jsonl"
SEED = "/data/prasann/ctc_seed_pools/outlier.seed.jsonl.gz"
h = lambda s: hashlib.sha1(s.strip().encode()).hexdigest()
need = {r["src_idx"] for r in ROWS}
src = {}
with open(SRC) as f:
    for i, line in enumerate(f):
        if i in need:
            src[i] = json.loads(line)
print("src rows", len(src), time.time() - t0, file=sys.stderr, flush=True)
want = set()
for ex in src.values():
    for d in ex["documents"]:
        want.add(h(d["text"]))
print("distinct passages", len(want), file=sys.stderr, flush=True)
title = {}
multi = 0
with gzip.open(SEED, "rt") as f:
    for ln, line in enumerate(f):
        d = json.loads(line)
        for t, bodies in d.get("articles", []):
            for b in bodies:
                k = h(b)
                if k in want:
                    if k in title and title[k] != t:
                        multi += 1
                    title.setdefault(k, t)
        del d
print("seed lines", ln + 1, "mapped", len(title), "multi-title", multi, time.time() - t0, file=sys.stderr, flush=True)
out, bad, nomap = {}, [], 0
for r in ROWS:
    ex = src[r["src_idx"]]
    docs = ex["documents"]
    assert len(docs) == r["n_docs"], (r["src_idx"], len(docs), r["n_docs"])
    assert sorted(ex["gold_doc_indices"]) == sorted(r["gold"]), (r["src_idx"], ex["gold_doc_indices"], r["gold"])
    ts = [title.get(h(d["text"])) for d in docs]
    nomap += sum(t is None for t in ts)
    gids, ids = [], {}
    for j, t in enumerate(ts):
        key = t if t is not None else f"__unmapped_{j}"
        gids.append(ids.setdefault(key, len(ids)))
    cnt = collections.Counter(gids)
    gg = {gids[g] for g in r["gold"]}
    ok = (len(gg) == 1 and cnt[next(iter(gg))] == 3 and all(c >= 4 for g, c in cnt.items() if g not in gg)
          and ts[r["gold"][0]] == ex["_meta"]["minority_label"])
    if not ok:
        bad.append({"src_idx": r["src_idx"], "counts": sorted(cnt.values()), "gold_groups": sorted(gg)})
    out[r["fingerprint"]] = gids
print("rows", len(out), "bad", len(bad), "unmapped docs", nomap, time.time() - t0, file=sys.stderr, flush=True)
for b in bad[:10]:
    print("BAD", b, file=sys.stderr)
sizes = collections.Counter(c for g in out.values() for c in collections.Counter(g).values())
print("group-size histogram", dict(sorted(sizes.items())), file=sys.stderr)
print(json.dumps({"groups": out, "bad": bad}))
