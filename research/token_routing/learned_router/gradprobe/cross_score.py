"""Cross-model gradient token importance: score tokens with one checkpoint (e.g. the pre-SFT base), drop and
measure answer CE with another (the finetuned checkpoint).

    python research/token_routing/learned_router/gradprobe/cross_score.py --task nq --rung 8k --rows 16 \\
        --ckpt /data/prasann/devloss_grid/ckpts/setA32k --ckpt-format olmo_st \\
        --score-ckpt /data/prasann/devloss_grid/ckpts/setAbase --score-ckpt-format olmo_st \\
        --data-root /data/prasann/devloss_grid/data --out ...

Hypothesis: a finetuned model is near-certain on the answer (CE ~ 0.001), so its gate gradient at g = 1 is
saturated noise; a weaker model (the base) still has a large answer CE and should have strong gradients at the
right tokens. Every rule below is scored on the same rows and HARD-dropped / evaluated with the --ckpt model
(grid semantics, ``Prober.hard_ce``); only the model that produced the score differs:
  base_a{X}_abs   |dCE/dg| of the SCORE model, one pass with every body gate at g = X (--score-alphas)
  base_a{X}_harm  signed predicted harm -dCE/dg of the score model at g = X (the base's loss is not saturated)
  self_a{X}_abs   the same from the eval model itself (--self-alphas; a0.5 is the record's best gold-blind rule)
  base_gold_ga / self_gold_ga   gold documents first, then fill by |grad| at g = 0.5 (oracle references)
  rand_s*, bar, gold_only       as in grad_importance.py
Every scored rule is also recorded at the bar's per-row budget (``<rule>@bar``). The answer is used to score, so
every rule except rand is an ORACLE, not deployable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grad_importance as GI  # noqa: E402

FR, TR, G, log = GI.FR, GI.TR, GI.G, GI.log


def load(path, fmt, vocab, ids, stop_ids, backend):
    if fmt == "olmo_st":
        m = GI.load_olmo_safetensors(path, vocab, ids, stop_ids, backend)
    else:
        m = G.load_model(path, fmt, vocab, ids, 42, stop_ids, backend)
    for p_ in m.parameters():
        p_.requires_grad_(False)
    m.eval()
    return m


def run_row(Pe, Ps, pe, ps, r, a, dev):
    """Pe / Ps: Prober for the eval / score model on the same row (same tokens, same body positions)."""
    out = {"T": pe["T"], "n_body": Pe.Nb, "gold_docs": sorted(r["gold"] or []), "ce_full": pe["ce_full"],
           "ce_full_score_model": ps["ce_full"], "checks": {}, "ce_alpha": {"score": {}, "self": {}}, "doc_level": {}}
    t0 = time.time()
    ones = torch.ones(Pe.Nb, device=dev)
    scores = {}
    for tag, P, alphas in (("base", Ps, a.score_alphas), ("self", Pe, a.self_alphas)):
        for al in [float(v) for v in alphas.split(",") if v]:
            ce_al, gz, _ = P.soft_ce_grad(torch.full_like(ones, al))
            out["ce_alpha"]["score" if tag == "base" else "self"][f"{al:g}"] = ce_al
            if al == 1.0:
                out["checks"][f"{tag}_soft_g1_vs_flash_full"] = ce_al - (ps if tag == "base" else pe)["ce_full"]
            scores[f"{tag}_a{al:g}_abs"] = gz.abs()
            if tag == "base":
                scores[f"{tag}_a{al:g}_harm"] = -gz
        if P.gold.any() and a.gold_curves:
            _, gh, _ = P.soft_ce_grad(torch.full_like(ones, 0.5))
            scores[f"{tag}_gold_ga"] = 2.0 * torch.tensor(P.gold, device=dev, dtype=torch.float32) + gh.abs() / gh.abs().max().clamp(min=1e-30)
    out["checks"]["grad_sec"] = time.time() - t0
    out["auroc_gold"] = {n: GI.auroc(s.cpu().numpy(), Pe.gold) for n, s in scores.items() if "gold_ga" not in n}
    out["auroc_gold"]["gold_token_frac"] = float(Pe.gold.mean())
    # rank agreement between the base and self scores (Spearman over body tokens)
    out["spearman_base_vs_self"] = {f"{b}~{s}": GI.spearman(scores[b].cpu().numpy(), scores[s].cpu().numpy())
                                    for b in scores if b.startswith("base_") and "gold" not in b
                                    for s in scores if s.startswith("self_") and "gold" not in s}

    keeps = sorted({float(k) for k in a.keeps.split(",")}, reverse=True)
    curves = {}

    def rec(name, rho, keep):
        ce, T2, fl = Pe.hard_ce(keep)
        curves.setdefault(name, []).append({"rho": rho, "k": int(keep.sum()), "ce": ce, "dce": ce - pe["ce_full"], "passes": 1,
                                            "T2": T2, "flops": fl, "gold_kept": float(keep.cpu().numpy()[Pe.gold].mean())
                                            if Pe.gold.any() else float("nan")})

    for rho in keeps:
        k = int(round(rho * Pe.Nb))
        for nm, sc in scores.items():
            rec(nm, rho, Pe.topk(sc, k))
        for s in range(a.rand_seeds):
            gen = torch.Generator().manual_seed(1000 * s + 7)
            rec(f"rand_s{s}", rho, Pe.topk(torch.rand(Pe.Nb, generator=gen).to(dev), k))
    bar_body = pe["real_bar"].to(dev)[Pe.body_idx]
    k_bar = int(bar_body.sum())
    for nm, sc in scores.items():
        rec(f"{nm}@bar", k_bar / max(Pe.Nb, 1), Pe.topk(sc, k_bar))
    rec("bar", float(bar_body.float().mean()), bar_body)
    if Pe.gold.any():
        rec("gold_only", float(Pe.gold.mean()), torch.tensor(Pe.gold, device=dev))
    out["curves"] = curves
    out["sec"] = time.time() - t0
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--row-offset", type=int, default=0)
    ap.add_argument("--ckpt", required=True, help="EVAL model: tokens are dropped and CE measured with this one")
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--score-ckpt", required=True, help="SCORE model: gradients are taken with this one")
    ap.add_argument("--score-ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--keeps", default="0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002")
    ap.add_argument("--score-alphas", default="1,0.75,0.5,0.25")
    ap.add_argument("--self-alphas", default="1,0.5")
    ap.add_argument("--gold-curves", type=int, default=1)
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--ckpt-blocks", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = TR.load_split(a.data_root, a.task, [a.rung], tok, ids, a.row_offset + a.rows)[a.row_offset:]
    log(f"{len(rows)} rows of {a.task}@{a.rung}, lens {[len(r['ids']) for r in rows]}")
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in rows])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    me = load(a.ckpt, a.ckpt_format, vocab, ids, stop_ids, a.attn_backend)
    ms = load(a.score_ckpt, a.score_ckpt_format, vocab, ids, stop_ids, a.attn_backend)
    freeze_fla_length_autotune()
    dev = next(me.parameters()).device
    TR.configure(me, stop_ids, tables.to(dev))
    TR.configure(ms, stop_ids, tables.to(dev))
    res = []
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for i, r in enumerate(rows):
        with torch.no_grad():
            pe = FR.base_prep(me, r, ids, dev)
            ps = FR.base_prep(ms, r, ids, dev)
        assert torch.equal(pe["x"], ps["x"]) and torch.equal(pe["feats"]["idx"], ps["feats"]["idx"]), "score/eval rows differ"
        ck = a.ckpt_blocks == "1" or (a.ckpt_blocks == "auto" and pe["T"] > 12000)
        Pe, Ps = GI.Prober(me, pe, dev, ckpt_blocks=ck), GI.Prober(ms, ps, dev, ckpt_blocks=ck)
        assert torch.equal(Pe.body_idx, Ps.body_idx)
        o = run_row(Pe, Ps, pe, ps, r, a, dev)
        res.append(o)
        el = time.time() - t0
        log(f"row {i + 1}/{len(rows)} T {o['T']} body {o['n_body']} CE_full eval {o['ce_full']:.4f} score {o['ce_full_score_model']:.4f} "
            f"| checks {json.dumps({k: round(v, 5) for k, v in o['checks'].items()})}")
        log("   AUROC gold: " + " ".join(f"{k} {v:.3f}" for k, v in o["auroc_gold"].items()))
        for n in sorted(o["curves"]):
            if not n.startswith("rand_s"):
                log(f"   {n:20s} " + "  ".join(f"{q['rho']:.3f}:{q['dce']:+.3f}" for q in o["curves"][n]))
        log(f"   (peak {o['peak_gb']:.1f} GB; {o['sec']:.0f}s this row; {el:.0f}s total, ETA {el / (i + 1) * (len(rows) - i - 1):.0f}s)")
        json.dump({"rows": res, "argv": sys.argv, "task": a.task, "rung": a.rung}, open(a.out, "w"), indent=1)
    summ = GI.summarize(res, a.keeps)
    summ["ce_full_score_model"] = float(np.mean([o["ce_full_score_model"] for o in res]))
    summ["spearman_base_vs_self"] = {k: float(np.nanmean([o["spearman_base_vs_self"][k] for o in res])) for k in res[0]["spearman_base_vs_self"]}
    log("SUMMARY " + json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": res, "argv": sys.argv, "task": a.task, "rung": a.rung}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
