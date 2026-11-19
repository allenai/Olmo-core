import sys, json, random
sys.path.insert(0, "research/token_routing/early_loss_proxy/random_doc_drop")
import build_ng_pools as B, relabel as R
from ctc.tasks.oolong.generate import build_example
pool = B.oolong_pool(); ctx = R.OolongCtx(pool)
sp = pool.for_split("train"); rng = random.Random(5)
ok = deg = 0
for i in range(60):
    ex = build_example(rng, corpus=sp, target_tokens=15500, min_tokens=12000)
    if ex is None: continue
    R.selfcheck("oolong", ex, ctx)
    n = R.n_units("oolong", ex)
    for _ in range(3):
        keep = sorted(rng.sample(range(n), max(4, int(0.2 * n))))
        try:
            s = R.subset("oolong", ex, keep, ctx); ok += 1
        except R.Degenerate: deg += 1
print("oolong selfcheck ok; subsets ok", ok, "degenerate", deg)
print(ex["queries"][0][:200], "->", ex["answers"], "| sub:", s["queries"][0][:200], "->", s["answers"], s["_meta"]["num_items"])
with open(B.GROUP_SRC) as f:
    for i, line in enumerate(f):
        if i >= 30: break
        ex = json.loads(line); R.selfcheck("grouping", ex)
s = R.grouping_subset(ex, range(0, 88, 3)); print(ex["queries"], s["queries"], s["answers"][0][:200])
exs = B.P.read_jsonl("/data/prasann/early_loss_proxy/v2build/gen/reorder_16k/reorder/train.jsonl", 20)
for ex in exs: R.selfcheck("reorder", ex)
s = R.reorder_subset(exs[0], [0, 5, 7, 30, 89]); print(exs[0]["gold_order"][:10], s["gold_order"], s["answers"])
