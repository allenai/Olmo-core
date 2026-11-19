"""Print one dropped row per task from ng16k and a variant / no-rewrite pool (row slot k): kept display ids, question
tail and the loss span, decoded. Run on the build node:  $PY show_rows.py ng16k_rddA_s0 ng16k_rddAnr_s0 [k]"""
import sys

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import build_pool_v2 as P  # noqa: E402

SH = "/data/prasann/early_loss_proxy/shards"
tok = Tokenizer.from_file(f"{P.TOKENIZER}/tokenizer.json")
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
pools = ["ng16k"] + sys.argv[1:3]
rows = {p: P.load_shard_rows(f"{SH}/{p}")[1] for p in pools}
for ti, t in enumerate(["reorder", "oolong", "grouping"]):
    i = ti * 1200 + k
    for p in pools:
        ids, m = rows[p][i]
        sp = P.doc_spans(ids)
        heads = [tok.decode([int(x) for x in ids[s + 1:min(e - 1, s + 12)]]).replace("\n", " ")[:28] for s, e in sp[:6]]
        a0 = int(np.flatnonzero(m)[0])
        q = tok.decode([int(x) for x in ids[max(sp[-1][1], a0 - 160):a0]]).replace("\n", " ")
        print(f"--- {t} slot {k} {p}: len {len(ids)} docs {len(sp)}\n  first doc heads: {heads}\n  ...{q[-260:]}\n"
              f"  ANSWER: {tok.decode([int(x) for x in ids[m]])[:300]!r}")
