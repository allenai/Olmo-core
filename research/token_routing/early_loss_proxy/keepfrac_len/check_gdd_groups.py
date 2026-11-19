"""CPU check of GoldDocDrop's --st-gdd-groups path on synthetic rows (96 docs: one 3-passage outlier group + 23 majority
groups): every kept majority group whole and >= 4, gold kept, unlisted rows unchanged. Point sys.path at a staged code dir."""
import sys, json, random, collections, tempfile
sys.path.insert(0, "/data/prasann/early_loss_proxy/code_38e4a1346/src")
import torch
from olmo_core.nn.gold_doc_drop import GoldDocDrop, GoldDocDropConfig
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row
DS, DE, EOS = 5, 6, 0
rng = random.Random(0)
def mkrow(nd):
    row = [9, 9]
    for d in range(nd):
        row += [DS] + [100 + d] * 3 + [DE]
    row += [7, 7]
    return row
nd = 96
S0 = len(mkrow(96)) + 4
rows, gtabs, groups = [], {}, {}
for b in range(4):
    sizes = [3] + [4] * 22 + [5]  # 96
    g = sum([[i] * s for i, s in enumerate(sizes)], [])
    rng.shuffle(g)
    r = mkrow(nd); r = [t + b * 1000 if t >= 100 else t for t in r]
    fp = content_fingerprint_from_row(r + [EOS] * (S0 - len(r)), EOS)
    gtabs[fp] = [d for d in range(nd) if g[d] == 0]; groups[fp] = g
    rows.append((r, g))
S = S0
ids = torch.tensor([r + [EOS] * (S - len(r)) for r, _ in rows])
class R: pass
for gp in ["", "groups"]:
    router = R(); router.gold = dict(gtabs); router.rank = 0
    p = ""
    if gp:
        p = tempfile.mktemp(suffix=".json"); json.dump({"groups": groups}, open(p, "w"))
    dd = GoldDocDrop(GoldDocDropConfig(keep_lo=0.05, keep_hi=0.5, groups_path=p, log_every=2), router=router,
                     doc_start_id=DS, doc_end_id=DE, eos_id=EOS)
    out, _ = dd.apply(ids, None, -100)
    for b, (r, g) in enumerate(rows):
        o = out[b].tolist()
        kept = sorted({(t - b * 1000 - 100) for t in o if t >= 100 + b * 1000 - 0 and t < 1000 * (b + 1) and t >= 100})
        cnt = collections.Counter(g[d] for d in kept)
        print(gp or "plain", "row", b, "kept docs", len(kept), "gold kept", cnt[0], "min majority group", min(c for k, c in cnt.items() if k != 0),
              "groups", len(cnt) - 1, "full groups", all(cnt[k] == g.count(k) for k in cnt))
# not-in-sidecar row passes whole
router = R(); router.gold = dict(gtabs); router.rank = 0
p = tempfile.mktemp(suffix=".json"); json.dump({}, open(p, "w"))
dd = GoldDocDrop(GoldDocDropConfig(keep_lo=0.05, keep_hi=0.5, groups_path=p, log_every=2), router=router, doc_start_id=DS, doc_end_id=DE, eos_id=EOS)
out, _ = dd.apply(ids, None, -100); print("unlisted unchanged", torch.equal(out, ids))
