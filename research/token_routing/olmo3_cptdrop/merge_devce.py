"""
Join row-chunk JSONs from eval_devce.py (--row-start / --rows) for ONE (checkpoint, shard) into the JSON a
single full run would have written: per_row concatenated in row order, summary recomputed, eval_size = all rows.
Refuses gaps, overlaps, or chunks that disagree on checkpoint / shard / yarn.

    python merge_devce.py --out D-480_dev32k_1024.json chunk_*.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eval_devce import summarize, summarize_ctx, summarize_repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("chunks", nargs="+")
    a = ap.parse_args()
    ds = sorted((json.load(open(p)) for p in a.chunks), key=lambda d: d["row_ids"][0])
    for k in ("ckpt", "dev", "yarn_factor", "row_len", "shard_meta", "family", "ctx_window"):
        vals = {json.dumps(d.get(k), sort_keys=True) for d in ds}
        if len(vals) != 1:
            raise SystemExit(f"chunks disagree on {k}: {vals}")
    ids = [i for d in ds for i in d["row_ids"]]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise SystemExit(f"row ids not contiguous/unique: first {ids[:3]} ... ({len(ids)} rows)")
    sums = np.concatenate([np.asarray(d["per_row"]["loss_sum"], dtype=np.float64) for d in ds])
    cnts = np.concatenate([np.asarray(d["per_row"]["tokens"], dtype=np.int64) for d in ds])
    bins = [(b["name"], b["lo"], b["hi"]) for b in ds[0]["bins"]]
    out = dict(ds[0])
    summ = summarize(sums, cnts, bins)
    per_row = {"loss_sum": sums.tolist(), "tokens": cnts.tolist()}
    if all("repeat_loss_sum" in d["per_row"] for d in ds):
        rs = np.concatenate([np.asarray(d["per_row"]["repeat_loss_sum"], dtype=np.float64) for d in ds])
        rc = np.concatenate([np.asarray(d["per_row"]["repeat_tokens"], dtype=np.int64) for d in ds])
        summ["repeat"] = summarize_repeat(rs, rc)
        per_row["repeat_loss_sum"], per_row["repeat_tokens"] = rs.tolist(), rc.tolist()
    if ds[0].get("ctx_window"):
        ctx = np.concatenate([np.asarray(d["per_row"]["ctx_full_win_cnt"], dtype=np.float64) for d in ds])
        summ["ctx_gain"] = summarize_ctx(ctx, ds[0]["ctx_window"])
        per_row["ctx_full_win_cnt"] = ctx.tolist()
    out.update(
        summary=summ, eval_size=len(ids), row_ids=ids,
        per_row=per_row,
        timing_s={"score_sum_over_chunks": sum(d["timing_s"]["score"] for d in ds)},
        merged_from=list(a.chunks),
    )
    json.dump(out, open(a.out, "w"), indent=1)
    o = out["summary"]["overall"]
    print(f"merged {len(ds)} chunks -> {a.out}: {len(ids)} rows, overall CE {o['ce']:.4f} | "
          + " ".join(f"{k} {v['ce']:.4f}" for k, v in out["summary"]["bins"].items())
          + (" | " + " ".join(f"{k} {v['ce']:.4f}" for k, v in summ["repeat"].items() if v["ce"] is not None) if "repeat" in summ else "")
          + (f" | ctx_gain {summ['ctx_gain']['gain']:+.4f} ±{summ['ctx_gain']['gain_row_se']:.4f}" if "ctx_gain" in summ else ""))


if __name__ == "__main__":
    main()
