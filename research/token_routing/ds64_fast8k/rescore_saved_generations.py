"""
Re-score SAVED docchunk-ladder generations with the grading added on 2026-10-25, without re-running
the model: reorder's parse-tolerant ``kendall_tau_tolerant`` / ``pairwise_acc_tolerant``, and (when
the generations exist) rerank's ctc ``ce_pos_recall``.

Input = the ladder's per-rung response files, ``<out>_<task>_<rung>_responses.jsonl`` (one row per
eval example: ``i``, ``response`` and the gold fields; written since 2026-10-20). For -lam runs they
sit on Lambda under ``/accounts/projects/sewonm/prasann/ctc_suite/eval_results/<run>/``; for Beaker
runs on weka under ``checkpoints/prasanns/<run>-ev<task>/eval/`` (copy them out first). Pass one
directory per train run, laid out as ``<root>/<train_run>/<task>_multirung_<task>_<rung>_responses.jsonl``.

reorder needs only the response file (it carries ``gold_order``). rerank's grader also needs each
row's ``ce_scores`` and documents: response files written before 2026-10-25 lack ``ce_scores``, so
pass ``--rows-root`` (the ``fetch_suite_eval_rows.py`` tree) and rows are joined by index ``i``.

As a guard the script also recomputes the STRICT primary metric and compares it with the value the
eval reported -- the ladder JSON next to the responses (``<task>_multirung.json``) when present, else
the ``ops result`` row for that (train run, task, rung); a mismatch beyond rounding skips that file
(the responses would not be the ones that were scored). New rows copy run_id / experiment / ckpt_path /
eval_job_id from the strict ops row, else from the ladder JSON's ``_meta`` (+ ``--experiment``,
``--cluster``). Stdlib + the vendored ``ctc`` package (no torch):

    python3 research/token_routing/ds64_fast8k/rescore_saved_generations.py \
        --root <dir> --runs <train_run>[,...] --tasks reorder --out rows.jsonl [--record]
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
LADDER = os.path.join(REPO, "recipes/eval/legacy_native/ctc_eval/eval/eval_lc_native_docchunk_ladder.py")
#: ladder task -> (ctc grader, the new metrics to record, suite subset)
TASKS = {
    "reorder": ("reorder", ("kendall_tau_tolerant", "pairwise_acc_tolerant"), "reorder"),
    "rerank": ("rerank", ("ce_pos_recall",), "rerank"),
}
RUNG_TOKENS = {"2k": 2048, "8k": 8192}


def _ladder():
    spec = importlib.util.spec_from_file_location("docchunk_ladder", LADDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ops_rows():
    out = subprocess.run([OPS, "result", "show", "--json"], capture_output=True, text=True, timeout=300).stdout
    return json.loads(out) if out.strip().startswith("[") else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="<root>/<train_run>/<task>_multirung_<task>_<rung>_responses.jsonl")
    ap.add_argument("--runs", required=True, help="comma list of train runs (subdirectories of --root)")
    ap.add_argument("--tasks", default="reorder", help=f"comma list from {sorted(TASKS)}")
    ap.add_argument("--rungs", default="2k,8k")
    ap.add_argument("--rows-root", default=None, help="suite rows tree, joined by index (needed for rerank)")
    ap.add_argument("--cluster", default="beaker", choices=["beaker", "lambda"],
                    help="where the eval ran (eval_backend label; lambda ckpt paths get a 'lambda:' prefix)")
    ap.add_argument("--experiment", default="tokdrop_lr", help="when no strict ops row names one")
    ap.add_argument("--out", required=True, help="JSONL of new result rows")
    ap.add_argument("--record", action="store_true", help="run `ops result add --json OUT --check`")
    a = ap.parse_args()

    L = _ladder()
    existing = _ops_rows()
    have = {(r.get("train_run"), r["task"], r["rung"], r.get("metric")) for r in existing}
    rows_out, problems = [], []
    for run in a.runs.split(","):
        for task in a.tasks.split(","):
            grader, metrics, subset = TASKS[task]
            spec = L.ctc_spec(grader)
            score_fn = L.make_ctc_score_fn(grader)
            for rung in a.rungs.split(","):
                path = os.path.join(a.root, run, f"{task}_multirung_{task}_{rung}_responses.jsonl")
                if not os.path.exists(path):
                    problems.append(f"{run} {task}@{rung}: no saved generations ({path})")
                    continue
                resp = [json.loads(line) for line in open(path)]
                resp.sort(key=lambda r: r["i"])
                if a.rows_root:
                    src = os.path.join(a.rows_root, subset, f"rung_{RUNG_TOKENS[rung]}.jsonl")
                    src_rows = [json.loads(line) for line in open(src)]
                    if len(src_rows) != len(resp):
                        problems.append(f"{run} {task}@{rung}: {len(resp)} responses vs {len(src_rows)} rows")
                        continue
                    examples = [{"ex": src_rows[r["i"]]} for r in resp]
                elif task == "reorder":
                    examples = [{"ex": {"gold_order": r["gold_order"],
                                        "documents": [None] * len(r["gold_order"])}} for r in resp]
                else:
                    problems.append(f"{run} {task}@{rung}: rerank needs --rows-root (ce_scores, documents)")
                    continue
                res, _ = score_fn(examples, [r.get("response") or "" for r in resp])
                strict = next((r for r in existing if r.get("train_run") == run and r["task"] == task
                               and r["rung"] == rung and r.get("metric") == spec.primary_metric), None)
                summ_p = os.path.join(a.root, run, f"{task}_multirung.json")
                summ = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
                smeta = summ.get("_meta") or {}
                ref = summ.get(f"{task}_{rung}", strict["value"] if strict else None)
                if task == "reorder":  # rerank had no strict number before: nothing to check
                    if ref is None:
                        problems.append(f"{run} {task}@{rung}: no reported strict {spec.primary_metric} to check against")
                        continue
                    if abs(ref - res[spec.primary_metric]) > 0.0015:
                        problems.append(f"{run} {task}@{rung}: recomputed strict {res[spec.primary_metric]:.4f} != "
                                        f"reported {ref} -- skipped")
                        continue
                ckpt = smeta.get("ckpt_path")
                meta = strict or {
                    "run_id": f"{run}-ev{task}", "experiment": a.experiment, "cot_label": smeta.get("cot_label"),
                    "eval_job_id": smeta.get("eval_job_id"),
                    "ckpt_path": (f"lambda:{ckpt}" if a.cluster == "lambda" and ckpt else ckpt),
                    "value": ref}
                print(f"{run} {task}@{rung}: eval_size={len(resp)} strict {spec.primary_metric}="
                      f"{res[spec.primary_metric]:.4f} (recorded {meta.get('value')}) | "
                      + " ".join(f"{k}={res[k]:.4f}" for k in metrics + ("id_coverage", "dup_rate", "parse_rate")
                                 if k in res), flush=True)
                for m in metrics:
                    if (run, task, rung, m) in have:
                        continue
                    rows_out.append(dict(
                        run_id=meta.get("run_id") or f"{run}-ev{task}", train_run=run,
                        experiment=meta.get("experiment"), task=task, rung=rung, metric=m,
                        value=round(float(res[m]), 4), eval_size=len(resp),
                        # tau / pairwise accuracy are not binomial proportions: no binomial SE
                        se=None, cot_label=meta.get("cot_label") or "no-cot",
                        eval_backend=f"olmo-core-native-docchunk-ladder ({a.cluster}; rescored from saved "
                                     "generations, 2026-10-25 grading)",
                        ckpt_path=meta.get("ckpt_path"), eval_job_id=meta.get("eval_job_id")))
    for p in problems:
        print(f"[!] {p}", flush=True)
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows_out))
    print(f"{len(rows_out)} new rows -> {a.out}")
    if a.record and rows_out:
        r = subprocess.run([OPS, "result", "add", "--json", a.out, "--check"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
    sys.exit(1 if problems and not rows_out else 0)


if __name__ == "__main__":
    main()
