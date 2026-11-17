"""Build reference.json (the numbers `repro.py reference` prints) from Prasann's ops results log.

Needs Prasann's ops toolkit (Berkeley machines only; optional for anyone else -- reference.json is committed).
Scores: `ops result show --json` (native docchunk ladder, eval_size 500 per cell, no-cot). PF: the last
`flop_meter/actual_pflops` of each training run's Beaker log (values copied below with their Beaker experiment ids;
dense fast8k f488/f976 ran on local slurm, PF from records/oracle-threshold-sweep.md). new4: results.json of the
2026-11-17 validation (research/token_routing/early_loss_proxy/validation/, branch loop/validate @ 787581912).

    python3 research/token_routing/recipe_repro/build_reference.py
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
PRIMARY = {"oolong": "score", "grouping": "pairwise_f1", "reorder": "kendall_tau"}

#: run -> (arm, budget, seed, PF, Beaker experiment of the training run)
FAST8K = {
    "f8k-mt4-dense-f488-tde-lr5e-06-cub": ("dense", "f488", 0, 379.0, "slurm (cubbins)"),
    "f8k-mt4-dense-f976-tde-lr5e-06-cub": ("dense", "f976", 0, 762.0, "slurm (cubbins)"),
    "f8k-mt4-dense-f1952-tde-lr5e-06": ("dense", "f1952", 0, 1531.25, "01M5QGZ7BXVERPKX365GRS0VH4"),
    "f8k-mt4-dense-f3904-tde-lr5e-06": ("dense", "f3904", 0, 3063.15, "01M5QH036NM7RNSPY6V56CCG2G"),
    "f8k-mt4-dense-f5856-tde-lr5e-06": ("dense", "f5856", 0, 4594.93, "01M5RX8TEC0VTPFG16BEPV6NXJ"),
    "f8k-mt4-dense-f7808-tde-lr5e-06": ("dense", "f7808", 0, 6128.22, "01M5RXA2XQPHWS71B23GA21NPZ"),
    "f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05": ("recipe", "f1952", 0, 345.33, "01M5VH5APW70Y48FW01Y7W3BVF"),
    "f8k-mt4-lora64all32r20to7sd1-f1952-tde-lr1e-05": ("recipe", "f1952", 1, 343.21, "01M6F3ZAR3PENM133YFNVHDTB5"),
    "f8k-mt4-lora64all32r20to7-f3904-tde-lr1e-05": ("recipe", "f3904", 0, 688.13, "01M5VH6FK7J23RHG272XB796M5"),
    "f8k-mt4-lora64all32r20to7-f7808-tde-lr1e-05": ("recipe", "f7808", 0, 1373.34, "01M5VH7FE3H2ER7VKEV2J50KBE"),
}
MT11 = {
    "m64-dense-f16-lr5e-06": ("dense", "f16", 0, 1175.18, "01M5KV26E5X06Z0DMN8D9S936N"),
    "m64-dense-f8-lr5e-06": ("dense", "f8", 0, 2287.33, "01M5KR0BQXMHGHHKXQ6QWZGJJR"),
    "m64-dense-f4-lr5e-06": ("dense", "f4", 0, 4624.72, "01M5M8PY3ZYDNHS7V27WWB2QWQ"),
    "m64-dense-f2-lr5e-06": ("dense", "f2", 0, 9294.35, "01M5NFVJS7Y4D046KXFE9AFE81"),
    "m64-dense-f1-lr5e-06": ("dense", "f1", 0, 18591.51, "01M5NRQYFBSQ2ZPME1X0MDSC2G"),
    "m64-lora64r20to7f16-f4-lr5e-06": ("recipe", "f4", 0, 1695.79, "01M5RG35YY70PX7XSAEYXWENPF"),
}
MT11_RUNGS = ("2k", "8k", "16k", "32k", "64k")


def ops_cells(rows, run, rungs):
    cells = {}
    for r in rows:
        if r.get("train_run") != run or r["rung"] not in rungs:
            continue
        if r["metric"] != PRIMARY.get(r["task"], "f1") or not r["run_id"].startswith(run):  # native ladder rows
            continue
        cells.setdefault(f"{r['task']}@{r['rung']}", [r["value"], r["eval_size"]])
    return cells


def main():
    rows = json.loads(subprocess.run([OPS, "result", "show", "--json"], capture_output=True, text=True).stdout)
    out = {}
    out["fast8k"] = {"source": "ops results (experiment tokdrop_lr), native ladder, eval_size 500/cell, 1 seed per "
                     "run (recipe f1952 has 2 seeds)",
                     "points": [dict(run=k, arm=a, budget=b, seed=s, pf=pf, beaker=ex,
                                     cells=ops_cells(rows, k, ("8k", "2k"))) for k, (a, b, s, pf, ex) in FAST8K.items()]}
    out["mt11"] = {"source": "ops results (experiment mt11_64k), native ladder 2k-64k on CTC-suite rows, eval_size "
                   "500/cell, 1 seed, LR 5e-6 for both arms",
                   "points": [dict(run=k, arm=a, budget=b, seed=s, pf=pf, beaker=ex,
                                   cells=ops_cells(rows, k, MT11_RUNGS)) for k, (a, b, s, pf, ex) in MT11.items()],
                   "notes": ["the recipe here is lora64r20to7f16 (LoRA on blocks 0-6 only, floor 16 tokens/doc), "
                             "not the whole-model-LoRA recipe", "dense f2/f1 and the recipe ran on 4 GPUs, dense "
                             "f16/f8/f4 on 8 (FLOPs do not depend on it)"]}
    res = json.load(open(os.path.join(HERE, "..", "early_loss_proxy", "validation", "results.json")))
    arm = {"denselora": "denselora", "recipe": "recipe", "dense": "dense"}
    pts = []
    for run, t in res["train"].items():
        a, b = run.split("-")[1], run.split("-")[2]
        cells = {k: [c["value"], c["eval_size"]] for k, c in res["cells"].get(run, {}).items()}
        pts.append(dict(run=run, arm=arm[a], budget=b, seed=1 if run.endswith("sd1") else 0, pf=t["pf"],
                        beaker=t["job"], cells=cells))
    out["new4"] = {"source": "early-loss-proxy validation 2026-11-17 (results.json, ops runs elpv-*), suite rows 4k/8k/16k, "
                   "eval_size 500/cell, 1 seed (recipe f1 2 seeds)", "points": pts,
                   "notes": ["reorder excluded from every mean: its strict-permutation grader scores every arm ~0 "
                             "(parse rate ~0.65)", "FTM is against the dense-LoRA curve (f4/f2/f1); dense full fine-tune "
                             "has one point (f1) and beats the recipe there"]}
    json.dump(out, open(os.path.join(HERE, "reference.json"), "w"), indent=1)
    for mix, v in out.items():
        print(mix, [(p["run"], len(p["cells"])) for p in v["points"]])


if __name__ == "__main__":
    main()
