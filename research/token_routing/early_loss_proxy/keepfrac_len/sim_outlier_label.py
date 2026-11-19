
"""Does random distractor drop keep the outlier label valid? Monte Carlo over the kl16k outlier rows' topic map.

Label valid = the outlier topic (3 gold passages, all kept) is the UNIQUELY smallest topic among the kept passages.
    python3 sim_outlier_label.py gdd_groups.json [gold_fingerprints.json]
"""
import collections, json, random, sys
g = json.load(open(sys.argv[1]))["groups"]
gold = json.load(open(sys.argv[2]))
rng = random.Random(0)
for lo, hi in ((0.05, 0.5), (0.05, 0.2), (0.25, 0.75)):
    n = valid = largest = strict = 0
    for fp, grp in g.items():
        if grp is None: continue
        gs = set(gold[fp]); og = grp[next(iter(gs))]
        for _ in range(20):
            r = rng.uniform(lo, hi); ng = [d for d in range(len(grp)) if d not in gs]; rng.shuffle(ng)
            keep = list(gs) + ng[:round(r * len(ng))]
            c = collections.Counter(grp[d] for d in keep); oth = [v for k, v in c.items() if k != og]
            n += 1; valid += all(v > c[og] for v in oth); largest += all(c[og] >= v for v in oth); strict += all(c[og] > v for v in oth)
    print(f"keep {lo:.2f}-{hi:.2f}: label valid {valid / n:.1%}, outlier topic is the strictly largest {strict / n:.1%}, largest incl. ties {largest / n:.1%} ({n} draws)")
