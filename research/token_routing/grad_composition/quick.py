"""Print the router-design numbers from results/<label>/probe.json + probe_stats.npz: parameter-gradient share in
blocks 0-6 vs 7-31, cross-layer rank correlation of token importance (block 7 vs 16 / 24), pyramid containment,
and how concentrated the importance of the tokens a 10% keep at block 7 drops is.

    python research/token_routing/grad_composition/quick.py base mt4-st122 ...
"""
import json
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    for lab in sys.argv[1:]:
        d = os.path.join(RES, lab)
        meta = json.load(open(os.path.join(d, "probe.json")))
        st = dict(np.load(os.path.join(d, "probe_stats.npz")))
        rows = meta["rows"]
        tasks = meta["tasks"]
        pyr = list(meta["pyr"])
        print(f"\n=== {lab} ({meta['ckpt_resolved']}); rows/task {meta['rows_per_task']} (eval_size flag: <500)")
        print(f"{'task':14s} {'n':>3s} {'CE':>6s} | pgrad energy 0-6 / 7-31 / head (row) | summed-grad 0-6 / 7-31 | "
              f"Spearman G7~G16 G7~G24 G16~G24 | top10%@7 holds G16^2 G24^2 | nested 50/10/3 vs one-shot@16 (G24^2) | "
              f"dropped share / top10% of dropped / eff-n | lag1")
        for t in tasks + ["all"]:
            idx = np.array([i for i, r in enumerate(rows) if t == "all" or r["task"] == t])
            if not len(idx):
                continue
            ce = np.mean([rows[i]["ce"] for i in idx])
            pg = st["pgrad"][idx]
            tot = pg.sum((1, 2))
            blk = pg[:, :32].sum(-1)
            a, b, h = (blk[:, :7].sum(1) / tot).mean(), (blk[:, 7:].sum(1) / tot).mean(), (pg[:, 32].sum(-1) / tot).mean()
            if t != "all":
                e = np.array(meta["task_summed_pgrad"][t])
                sa, sb = e[:7].sum() / e.sum(), e[7:32].sum() / e.sum()
            else:
                e = sum(np.array(v) for v in meta["task_summed_pgrad"].values())
                sa, sb = e[:7].sum() / e.sum(), e[7:32].sum() / e.sum()
            xc = st["xcorr/g"][idx].mean(0)
            cm = st["pyr_contain_mass"][idx].mean(0)  # (a, b, ka)
            i7, i16, i24 = pyr.index(7), pyr.index(16), pyr.index(24)
            ka10 = list(meta["pyr_ka"]).index(0.1)
            nest = st["pyr_nested"][idx].mean(0)
            sets, tg = list(meta["pyr_sets"]), list(meta["pyr_tgts"])
            dc = st["drop_conc"][idx].mean(0)
            print(f"{t:14s} {len(idx):3d} {ce:6.3f} | {a:.3f} / {b:.3f} / {h:.3f} | {sa:.3f} / {sb:.3f} | "
                  f"{xc[7, 16]:+.2f} {xc[7, 24]:+.2f} {xc[16, 24]:+.2f} | {cm[i7, i16, ka10]:.2f} {cm[i7, i24, ka10]:.2f} | "
                  f"{nest[sets.index('nest_3_7_16'), tg.index('g24')]:.2f} vs {nest[sets.index('one_16_3pct'), tg.index('g24')]:.2f} "
                  f"(one@7 3%: {nest[sets.index('one_7_3pct'), tg.index('g24')]:.2f}, rnd {nest[sets.index('random_3pct'), tg.index('g24')]:.2f}) | "
                  f"{dc[0]:.2f} / {dc[1]:.2f} / {dc[3]:.2f} | {dc[8]:+.2f}")


if __name__ == "__main__":
    main()
