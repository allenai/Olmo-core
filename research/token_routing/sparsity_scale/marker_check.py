"""Marker check for the sparsity_scale sweep (records/sparsity-scale-plan.md).

The fs35 dense checkpoints trained on marker-free shards, but the drop/skip harnesses find documents
through the <|box_start|>/<|box_end|> markers. Per eval row, this renders the SAME prompt twice with
the converter's own ``tokenize_example`` (``doc_markers`` True / False: the only difference is 2 marker
tokens per document) and records teacher-forced answer CE on the frozen checkpoint. Gate: the paired
mean dCE (markers - none) stays small, else the sweep cannot use marker-wrapped rows on these models.

    python research/token_routing/sparsity_scale/marker_check.py --size 0.8b --task contradiction \\
        --ckpt /weka/.../ckpts/fs35s08b-contradiction-dense-s56M --eval-jsonl /weka/.../..._n190_k3.jsonl \\
        --rows 128 --out /results/marker_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "recipes", "data"))

FACTORY = {"0.8b": "qwen3_5_0_8B", "2b": "qwen3_5_2B", "4b": "qwen3_5_4B", "9b": "qwen3_5_9B", "27b": "qwen3_5_27B"}
VOCAB = 248320
# converter settings of the taskscale dense arms (build_kv35_shards_beaker.sh re-tokenized the same JSONL)
CHUNK_BY = {"contradiction": "document", "oolong": "line"}


def log(m: str) -> None:
    print(f"[marker_check] {m}", flush=True)


def find_distcp(root: str) -> str:
    if root.rstrip("/").endswith("model_and_optim"):
        return root
    cands = sorted(glob.glob(f"{root}/model_and_optim") + glob.glob(f"{root}/step*/model_and_optim"),
                   key=lambda p: int(p.split("/step")[-1].split("/")[0]) if "/step" in p else -1)
    if not cands:
        raise SystemExit(f"no model_and_optim under {root}")
    return cands[-1]


def build_model(size: str, ckpt: str, attn_backend: str):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    cfg = getattr(TransformerConfig, FACTORY[size])(vocab_size=VOCAB, attn_backend=AttentionBackendName(attn_backend))
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = find_distcp(ckpt)
    t0 = time.time()
    load_model_and_optim_state(ck, model)
    log(f"loaded {ck} in {time.time() - t0:.0f}s ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")
    model = model.to(torch.bfloat16).cuda().eval()  # cast on CPU first: 27B fp32 does not fit an 80 GB GPU
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def answer_ce(model, ids: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Mean teacher-forced CE over the answer tokens (labels_mask True); position t predicts t + 1."""
    x = torch.as_tensor(ids.astype(np.int64), device="cuda")[None]
    tgt_pos = np.nonzero(mask)[0]
    tgt_pos = tgt_pos[tgt_pos > 0]
    pred = torch.as_tensor(tgt_pos - 1, device="cuda")
    lg = model(x, logits_to_keep=pred[None])[0].float()
    return float(F.cross_entropy(lg, x[0, pred + 1])), int(len(tgt_pos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=sorted(FACTORY))
    ap.add_argument("--task", required=True, choices=sorted(CHUNK_BY))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--query-position", default="after")
    ap.add_argument("--cot-mode", default="none")
    ap.add_argument("--use-titles", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from convert_unified_to_document_landmark import tokenize_example
    from olmo_core.data.document_chunk_landmark import RESERVED_IDS

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids_set = RESERVED_IDS["qwen3_5"]
    kw = dict(task=a.task, emit="dense", query_position=a.query_position, cot_mode=a.cot_mode, mem_freq=63,
              seq_len=1 << 20, chunk_by=CHUNK_BY[a.task], item_regex=r"\|\|", use_titles=a.use_titles, ids_set=ids_set)
    examples = [json.loads(l) for l in open(a.eval_jsonl)][: a.rows]
    log(f"{len(examples)} rows from {a.eval_jsonl}")

    model = build_model(a.size, a.ckpt, a.attn_backend)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows, t0 = [], time.time()
    for i, ex in enumerate(examples):
        mk = tokenize_example(tok, ex, doc_markers=True, **kw)
        nm = tokenize_example(tok, ex, doc_markers=False, **kw)
        if mk is None or nm is None:
            log(f"row {i}: converter skipped it")
            continue
        n_mk = int(np.isin(mk[0], [ids_set.doc_start, ids_set.doc_end]).sum())
        # the answer tokens must be identical. The lengths differ by the marker count for chunk_by=document;
        # for chunk_by=line (oolong) wrapping a line also changes how the adjacent newlines merge, so they differ more
        assert mk[0][mk[1]].tolist() == nm[0][nm[1]].tolist(), "answer spans differ"
        ce_mk, n_ans = answer_ce(model, *mk)
        ce_nm, _ = answer_ce(model, *nm)
        rows.append({"i": i, "T_markers": len(mk[0]), "T_none": len(nm[0]), "n_markers": n_mk, "len_diff_minus_markers": len(mk[0]) - len(nm[0]) - n_mk, "n_answer": n_ans,
                     "ce_markers": ce_mk, "ce_none": ce_nm, "dce": ce_mk - ce_nm})
        if i + 1 in (1, 2, 5, 10) or (i + 1) % 25 == 0:
            el = time.time() - t0
            log(f"row {i + 1}/{len(examples)} T {len(nm[0])} markers {n_mk} ce_none {ce_nm:.4f} ce_markers {ce_mk:.4f} "
                f"({el:.0f}s, ETA {el / (i + 1) * (len(examples) - i - 1):.0f}s)")
    d = np.array([r["dce"] for r in rows])
    summ = {"size": a.size, "task": a.task, "ckpt": a.ckpt, "eval_jsonl": a.eval_jsonl, "eval_size": len(rows),
            "ce_none_mean": float(np.mean([r["ce_none"] for r in rows])),
            "ce_markers_mean": float(np.mean([r["ce_markers"] for r in rows])),
            "dce_mean": float(d.mean()), "dce_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "dce_median": float(np.median(d)), "dce_abs_p90": float(np.quantile(np.abs(d), 0.9)),
            "mean_T_none": float(np.mean([r["T_none"] for r in rows])), "mean_markers": float(np.mean([r["n_markers"] for r in rows])),
            "converter": {k: v for k, v in kw.items() if k != "ids_set"}, "argv": sys.argv}
    log("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k != "argv"}))
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
