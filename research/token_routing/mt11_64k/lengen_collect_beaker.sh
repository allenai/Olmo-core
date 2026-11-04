#!/bin/bash
# CPU gantry job: print every olmo-eval CTC job result (metrics.json) under _olmoeval_ctc/<GLOB> as one line each,
# so lengen_pool.py can pool them on the login node (weka is not mounted there).
#   GLOB='m11lg-*' bash research/token_routing/mt11_64k/lengen_collect_beaker.sh   -> prints the Beaker experiment id
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
  --env GLOB="$GLOB" --allow-dirty --timeout 0 --yes -- /opt/conda/bin/python -c '
import glob, json, os
root = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_olmoeval_ctc"
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
        # empty generations: any *.jsonl prediction file beside metrics.json
        for p in glob.glob(os.path.join(os.path.dirname(m), "**", "*.jsonl"), recursive=True):
            n = e = 0
            try:
                for line in open(p):
                    r = json.loads(line); n += 1
                    g = r.get("generation") or r.get("prediction") or r.get("output") or r.get("response")
                    if isinstance(g, list): g = g[0] if g else ""
                    if isinstance(g, dict): g = g.get("text", "")
                    if not str(g or "").strip(): e += 1
            except Exception:
                continue
            print(f"LGEMPTY\t{run}\t{job}\t{os.path.relpath(p, d)}\t{n}\t{e}", flush=True)
print("LGDONE")
' 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1
