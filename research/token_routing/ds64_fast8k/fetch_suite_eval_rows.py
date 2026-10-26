"""
Fetch CTC-suite eval rows from the HF dataset into the unified-JSONL layout the native docchunk
ladder reads (``eval_lc_native_docchunk_ladder.py`` via ``load_unified_examples``).

Writes ``<out>/suite/<subset>/rung_<tokens>.jsonl`` (the same naming olmo-eval's
``CTC_SUITE_DATA_ROOT`` uses, incl. ``contradiction_iid``'s r2k -> 2560 alias) plus
``<out>/suite/MANIFEST.json`` (dataset revision, rows and sha256 per file). Rows are the first
``--rows`` of each parquet, read at the revision olmo-eval pins (``HF_REVISION``), so the ladder
grades the same rows olmo-eval does. Each row is the parquet record as-is (as olmo-eval and
``recipes/eval/setA_audit`` read it), except that ``meta`` / ``_meta`` stored as JSON strings are
decoded back to dicts.

Small I/O, stdlib + huggingface_hub + pyarrow; fine on a login node:

    python3 research/token_routing/ds64_fast8k/fetch_suite_eval_rows.py --out /data/prasann/tdlr_eval

``--verify ROOT`` checks an existing ``<subset>/rung_*.jsonl`` tree instead of fetching: every file in
the reference manifest (``--manifest``, default the checked-in ``suite_rows_MANIFEST.json``, an
independent fetch at the pinned revision) must exist with the same row count and sha256. Exit 1 on any
mismatch. The Beaker staging job (``stage_suite_rows_beaker.sh``) runs it on the weka copy.
"""

import argparse
import hashlib
import json
import os

HF_DATASET = "PrasannSinghal/ctc-suite-eval"
#: olmo-eval ctc_suite.HF_REVISION (prasann/ctc-suite-flopdrop): the commit every suite number reads.
HF_REVISION = "04ab86006124b3f34c566f44bbc82d4fec98fcda"
#: the six setA tasks the ladder grades through the ctc spec (ladder task == HF subset, except none)
NEW_SUBSETS = ["hotpotqa", "qdmatch_nq", "xabsence", "reorder", "strmatch", "grouping"]
#: the suite rows of the base ladder tasks graded through the ctc spec (--ctc-grade); rerank added
#: 2026-10-25 (it is in the mt11 training mix but had no suite rows staged, so it was never scored)
EXISTING_SUBSETS = ["nq", "outlier", "oolong", "contradiction_iid", "rerank"]
RUNG_TOKENS = {"r2k": 2048, "r4k": 4096, "r8k": 8192, "r16k": 16384, "r32k": 32768}
#: olmo-eval ROSTER rung_alias: the IID contradiction ladder starts one step up at r2k
RUNG_ALIAS = {("contradiction_iid", "r2k"): 2560}
#: reference manifest of the default fetch (11 subsets x r2k/r8k x 500 rows at HF_REVISION)
REF_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suite_rows_MANIFEST.json")


def verify(root: str, manifest: str) -> int:
    """Compare ``root``'s rung files with ``manifest`` (rows + sha256); returns the mismatch count."""
    ref = json.load(open(manifest))
    bad = 0
    for rel, want in sorted(ref["files"].items()):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"MISSING {rel}")
            bad += 1
            continue
        data = open(path, "rb").read()
        rows, sha = data.count(b"\n"), hashlib.sha256(data).hexdigest()
        ok = rows == want["rows"] and sha == want["sha256"]
        bad += not ok
        print(f"{'ok' if ok else 'MISMATCH'} {rel}: rows={rows} (want {want['rows']}) "
              f"sha256={sha[:12]} (want {want['sha256'][:12]})")
    print(f"verify {root} vs {manifest} (revision {ref.get('revision')}): "
          f"{len(ref['files']) - bad}/{len(ref['files'])} files match")
    return bad


def _normalize(ex: dict) -> dict:
    """The HF parquet stores dict-valued fields (outlier ``meta``, ``_meta``) as JSON strings."""
    for key in ("meta", "_meta"):
        if isinstance(ex.get(key), str):
            try:
                ex[key] = json.loads(ex[key])
            except ValueError:
                pass
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output root; files land under <out>/suite/")
    ap.add_argument("--verify", default=None, metavar="ROOT",
                    help="check ROOT/<subset>/rung_*.jsonl against --manifest instead of fetching")
    ap.add_argument("--manifest", default=REF_MANIFEST, help="reference manifest for --verify")
    ap.add_argument("--subsets", nargs="+", default=NEW_SUBSETS + EXISTING_SUBSETS)
    ap.add_argument("--rungs", nargs="+", default=["r2k", "r8k"])
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--revision", default=HF_REVISION)
    ap.add_argument("--cache", default=None, help="parquet download dir (default <out>/.hf_parquet)")
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(1 if verify(args.verify, args.manifest) else 0)
    if not args.out:
        ap.error("--out is required unless --verify is given")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cache = args.cache or os.path.join(args.out, ".hf_parquet")
    root = os.path.join(args.out, "suite")
    manifest = {"dataset": HF_DATASET, "revision": args.revision, "rows_requested": args.rows,
                "files": {}}
    for sub in args.subsets:
        for rung in args.rungs:
            tokens = RUNG_ALIAS.get((sub, rung), RUNG_TOKENS[rung])
            pq_path = hf_hub_download(HF_DATASET, f"data/{sub}/{rung}.parquet", repo_type="dataset",
                                      revision=args.revision, local_dir=cache)
            pf = pq.ParquetFile(pq_path)
            rows = []
            for batch in pf.iter_batches(batch_size=min(args.rows, 256)):
                rows.extend(batch.to_pylist())
                if len(rows) >= args.rows:
                    break
            rows = [_normalize(r) for r in rows[: args.rows]]
            os.makedirs(os.path.join(root, sub), exist_ok=True)
            out = os.path.join(root, sub, f"rung_{tokens}.jsonl")
            with open(out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
            rel = os.path.relpath(out, root)
            manifest["files"][rel] = {"subset": sub, "rung": rung, "rows": len(rows),
                                      "parquet_rows": pf.metadata.num_rows, "sha256": sha}
            flag = "" if len(rows) >= 500 else f"  (eval_size={len(rows)} < 500)"
            print(f"{rel}: {len(rows)} rows of {pf.metadata.num_rows}{flag}", flush=True)
    with open(os.path.join(root, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest['files'])} files under {root}")


if __name__ == "__main__":
    main()
