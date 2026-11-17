"""Shortcut check for doc-drop runs (records/docdrop-nongold-baseline.md). Stdlib; reads the eval response files.

Training with every gold document kept raises the gold share of the visible documents several-fold (contradiction
3% -> 18%, nq 2% -> 13%, outlier 5% -> 27%). A model that learned "what is present is likely the answer" would
over-predict on the full-context eval. Per (run, task, rung) this reports: empty responses, responses with no
parseable ID, mean predicted vs gold item count (contradiction pairs, outlier IDs), precision and recall of the
predicted items, and for oolong the empty share.

    python3 research/token_routing/docdrop/shortcut_check.py --runs <run>[,<run>...] [--node mooney] [--json out.json]
"""
import argparse
import json
import os
import re

TASKS = {"contradiction": "contra_multirung_contradiction", "nq": "nq_multirung_nq",
         "outlier": "outlier_multirung_outlier", "oolong": "oolong_multirung_oolong"}


def items(task, text):
    if task == "contradiction":
        pairs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        return {tuple(sorted((int(a), int(b)))) for a, b in pairs}
    if task == "outlier":
        text = text.split("Outliers:", 1)[1] if "Outliers:" in text else ""
    return {int(x) for x in re.findall(r"\d+", text)}


def gold_items(task, rec):
    g = rec.get("gold_doc_indices")
    if task == "contradiction":
        return {tuple(sorted(int(x) for x in p)) for p in g}
    if task == "outlier":  # gold_doc_indices are 0-based, the answer prints 1-based IDs
        return {int(x) + 1 for x in g}
    return None


def check(path, task):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    empty = sum(1 for r in rows if not str(r.get("response", "")).strip())
    out = {"eval_size": n, "empty": empty / n}
    if task == "oolong":
        return out
    pred = [items(task, str(r["response"])) for r in rows]
    out["unparsed"] = sum(1 for p in pred if not p) / n
    if task == "nq":
        out["mean_pred_items"] = sum(len(p) for p in pred) / n
        return out
    gold = [gold_items(task, r) for r in rows]
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(len(p) for p in pred), sum(len(g) for g in gold)
    out.update({"mean_pred_items": npred / n, "mean_gold_items": ngold / n,
                "precision": tp / max(1, npred), "recall": tp / max(1, ngold),
                "over_predict_rows": sum(1 for p, g in zip(pred, gold) if len(p) > len(g)) / n,
                "under_predict_rows": sum(1 for p, g in zip(pred, gold) if len(p) < len(g)) / n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--node", default="mooney")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for run in a.runs.split(","):
        base = f"/net/{a.node}/data/prasann/f8k_eval/runs/{run}/eval"
        for task, stem in TASKS.items():
            for rung in ("8k", "2k"):
                p = f"{base}/{stem}_{rung}_responses.jsonl"
                if not os.path.exists(p):  # a finished cell before the job's final copy into eval/
                    p = f"{os.path.dirname(base)}/cells/{task}_{rung}/{stem}_{rung}_responses.jsonl"
                if os.path.exists(p):
                    res[f"{run}|{task}|{rung}"] = check(p, task)
    for k, v in res.items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
