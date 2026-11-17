#!/bin/bash
# CPU gantry job: print every olmo-eval CTC job result (metrics.json) under _olmoeval_ctc/<GLOB> as one line each,
# so lengen_pool.py can pool them on the login node (weka is not mounted there).
#   GLOB='m11lg-*' bash research/token_routing/mt11_64k/lengen_collect_beaker.sh   -> prints the Beaker experiment id
# GANTRY_REF=<pushed sha>: run from that commit (the code is inline; for when the local tip is not pushed).
# Lines: LGM<TAB><run><TAB><job><TAB><json tasks>; plus LGEMPTY lines (empty-generation counts per task) when the
# job directory holds prediction files.
set -euo pipefail
GLOB="${GLOB:-m11lg-*}"
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
"$GANTRY" run --name "mt11-lengen-collect-$(date +%m%d%H%M%S)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 2 --memory 8GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  ${GANTRY_REF:+--ref $GANTRY_REF} --env GLOB="$GLOB" --allow-dirty --timeout 0 --yes -- /opt/conda/bin/python -c '
import glob, json, os
root = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_olmoeval_ctc"
shown = set()
for d in sorted(glob.glob(os.path.join(root, os.environ["GLOB"]))):
    run = os.path.basename(d)
    for m in sorted(glob.glob(os.path.join(d, "job*", "metrics.json"))):
        job = os.path.basename(os.path.dirname(m))
        try:
            tasks = json.load(open(m)).get("tasks", [])
        except Exception as e:
            print(f"LGERR\t{run}\t{job}\t{e}"); continue
        slim = [{k: t.get(k) for k in ("task", "primary_metric", "metrics", "num_instances")} for t in tasks]
        s = json.dumps(slim)
        print("LGM\t%s\t%s\t%s" % (run, job, s), flush=True)
        # empty generations: the *-predictions.jsonl files beside metrics.json (keys of the first record printed once)
        for p in glob.glob(os.path.join(os.path.dirname(m), "predictions", "**", "*-predictions.jsonl"), recursive=True):
            n = e = 0
            def gen_text(r):
                for k in ("generation", "generations", "prediction", "predictions", "completion", "completions",
                          "output", "outputs", "response", "responses", "model_output", "text"):
                    if k in r:
                        g = r[k]
                        while isinstance(g, list):
                            g = g[0] if g else ""
                        if isinstance(g, dict):
                            g = g.get("text") or g.get("generation") or g.get("completion") or ""
                        return str(g or "")
                return None
            try:
                for i, line in enumerate(open(p)):
                    r = json.loads(line); n += 1
                    if i == 0 and run not in shown:
                        shown.add(run)
                        print("LGKEYS\t%s\t%s" % (run, json.dumps({k: str(v)[:120] for k, v in r.items()})), flush=True)
                    g = gen_text(r)
                    if g is not None and not g.strip():
                        e += 1
            except Exception:
                continue
            print("LGEMPTY\t%s\t%s\t%s\t%d\t%d" % (run, job, os.path.basename(p), n, e), flush=True)
print("LGDONE")
' 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1
