#!/usr/bin/env python3
"""Delete ONLY the literal paths in a manifest, after re-checking each one on weka.

Runs inside a 0-GPU Beaker job with weka mounted (launch_weka_apply.py ships it inline).
Manifest lines are `<mode>\t<absolute path>`; no globs, no discovery. Modes:

  wholedir  a whole run / export dir. Must hold a checkpoint (a .metadata, *.distcp or
            *.safetensors file) and have no file modified in the last --fresh-days.
  step      a stepN dir. The run's highest stepN must survive, must not itself be listed,
            and must hold a .metadata. The run must be idle for --fresh-min minutes.
  stepmo    a stepN dir in a run that also holds a non-empty model_and_optim/.metadata
            (the model-only final save). Run idle for --fresh-min minutes.
  copydir   olmo3_cptdrop/<arm>/{base,data}: per-run copies of what is on S3 _transfer/.

Checks for every path: under ROOT, no '..', a real directory (not a symlink), not protected.
All verdicts are computed in a pre-pass BEFORE any delete (deleting a child bumps the parent's
mtime, which would otherwise trip the freshness guard). Dry run unless --apply.
"""
import argparse, json, os, re, shutil, sys, time

ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
ROOT = os.environ.get("WEKA_APPLY_TEST_ROOT", ROOT)  # local fixture tests only; never set on weka
PROTECT = [r"/ctc_suite/bases(/|$)", r"/ctc_olmo3/bases(/|$)", r"xlong5_2k256k_qwen35",
           r"contra_mix_.*_10k_128k", r"_eval_bundle", r"ctc-olmo3-7b-hpqaret-", r"vllm_serving",
           r"/shards(/|$)", r"_inventory", r"_olmoeval_ctc", r"_eval_results", r"cr_suite_data",
           r"ctc_sft_sets", r"hf_models"]
CKPT_FILE = re.compile(r"(^\.metadata$|\.distcp$|\.safetensors$)")


def scan(path, fresh_cut):
    """One walk: total bytes, newest mtime, whether it holds checkpoint files."""
    size, newest, ckpt = 0, 0.0, False
    for dp, dns, fns in os.walk(path, followlinks=False):
        for fn in fns:
            try:
                st = os.lstat(os.path.join(dp, fn))
            except OSError:
                continue
            size += st.st_size
            newest = max(newest, st.st_mtime)
            ckpt = ckpt or bool(CKPT_FILE.search(fn))
    return size, newest, ckpt


def steps_of(run):
    out = {}
    for n in os.listdir(run):
        m = re.fullmatch(r"step(\d+)", n)
        if m and os.path.isdir(os.path.join(run, n)):
            out[int(m.group(1))] = os.path.join(run, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fresh-days", type=float, default=7.0)
    ap.add_argument("--fresh-min", type=float, default=90.0)
    a = ap.parse_args()
    now = time.time()
    rows = [l.rstrip("\n").split("\t") for l in open(a.manifest) if l.strip()]
    listed = {p for _, p in rows}
    res, idle_run = [], {}
    for i, (mode, p) in enumerate(rows):
        why, size = "", 0
        if mode not in ("wholedir", "step", "stepmo", "copydir"):
            why = f"bad mode {mode}"
        elif not p.startswith(ROOT + "/") or ".." in p:
            why = "outside namespace / traversal"
        elif any(re.search(x, p) for x in PROTECT):
            why = "protected"
        elif os.path.islink(p):
            why = "symlink"
        elif not os.path.isdir(p):
            why = "missing (already gone?)"
        if not why and mode == "wholedir":
            size, newest, ckpt = scan(p, now)
            if not ckpt:
                why = "no checkpoint files inside"
            elif now - newest < a.fresh_days * 86400:
                why = f"file modified {(now - newest) / 86400:.1f}d ago"
        elif not why and mode == "step":
            run = os.path.dirname(p)
            st = steps_of(run)
            top = max(st) if st else None
            if top is None or st[top] == p:
                why = "is the run's final step"
            elif st[top] in listed:
                why = "final step is itself listed"
            elif not os.path.isfile(os.path.join(st[top], ".metadata")) and not any(
                    os.path.isfile(os.path.join(st[top], s, ".metadata")) for s in ("model_and_optim",)):
                why = "final step has no .metadata"
            else:
                if run not in idle_run:
                    _, newest, _ = scan(run, now)
                    idle_run[run] = now - newest
                if idle_run[run] < a.fresh_min * 60:
                    why = f"run modified {idle_run[run] / 60:.0f}m ago"
                else:
                    size, _, _ = scan(p, now)
        elif not why and mode == "stepmo":
            # stepN in a run that also holds the model-only final save: weights survive in it.
            run = os.path.dirname(p)
            mo = os.path.join(run, "model_and_optim", ".metadata")
            if not re.fullmatch(r"step\d+", os.path.basename(p)):
                why = "not a step<N> dir"
            elif not (os.path.isfile(mo) and os.path.getsize(mo) > 0):
                why = "run has no model-only final (model_and_optim/.metadata)"
            else:
                if run not in idle_run:
                    _, newest, _ = scan(run, now)
                    idle_run[run] = now - newest
                if idle_run[run] < a.fresh_min * 60:
                    why = f"run modified {idle_run[run] / 60:.0f}m ago"
                else:
                    size, _, _ = scan(p, now)
        elif not why and mode == "copydir":
            if not re.fullmatch(re.escape(ROOT) + r"/olmo3_cptdrop/[^/]+/(base|data)", p):
                why = "copydir outside olmo3_cptdrop/<arm>/{base,data}"
            else:
                size, newest, _ = scan(os.path.dirname(p), now)
                if now - newest < 86400:
                    why = "arm dir modified <1d ago"
                else:
                    size, _, _ = scan(p, now)
        res.append(dict(mode=mode, path=p, bytes=size, verdict="REFUSE" if why else "OK", why=why))
        if i < 3 or i % 100 == 0:
            print(f"[verify {i + 1}/{len(rows)}] {res[-1]['verdict']} {why} {p}", flush=True)
    ok = [r for r in res if r["verdict"] == "OK"]
    print(f"VERIFIED {len(ok)} ok / {len(res) - len(ok)} refused; "
          f"{sum(r['bytes'] for r in ok) / 2**40:.2f} TiB reclaimable", flush=True)
    for r in res:
        if r["verdict"] != "OK":
            print(f"REFUSE [{r['why']}] {r['path']}", flush=True)
    if a.apply:
        t0 = time.time()
        for j, r in enumerate(ok):
            try:
                shutil.rmtree(r["path"])
                r["deleted"] = True
            except Exception as e:  # keep going; report
                r["deleted"], r["error"] = False, repr(e)
            if j < 3 or j % 50 == 0:
                el = time.time() - t0
                print(f"[delete {j + 1}/{len(ok)}] {r['path']} ({el:.0f}s, "
                      f"ETA {el / (j + 1) * (len(ok) - j - 1):.0f}s)", flush=True)
        print(f"DELETED {sum(1 for r in ok if r.get('deleted'))} / {len(ok)}; "
              f"{sum(r['bytes'] for r in ok if r.get('deleted')) / 2**40:.2f} TiB", flush=True)
    json.dump(dict(apply=a.apply, generated=now, results=res), open(a.out, "w"), indent=0)


if __name__ == "__main__":
    sys.exit(main())
