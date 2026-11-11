"""HF reference CE of a multimodal-SFT step-1 global batch (gate (a): step-1 CE within 1e-3).

open-instruct's ``scripts/c12/h161_refce.py`` method: bf16 weights, each document of a packed row
forwarded as its own sequence, fp32 logits and CE, token-weighted over the loss tokens of all
rows. Reads the ``rankNN.npz`` that the trainer's ``BatchHashCallback(step1_reference=True)``
saves (``input_ids``, ``loss_masks`` where position i predicts token i + 1, ``document_ids``).

    python olmo35_mm_sft_refce.py DUMP_DIR HF_DIR OUT.json [--seed-think-rows]

``HF_DIR`` may be a text export (``Olmo3MoeForCausalLM``) or a multimodal one (its
``language_model`` is used, text only). ``--seed-think-rows`` sets the ``<think>``/``</think>``
rows (input embedding and LM head) to the mean of their old pieces first, as the trainer does
before step 1, for exports made before seeding.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

THINK_ROWS = [(100256, [14023, 771, 29]), (100266, [524, 27963, 29])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("hf_dir")
    p.add_argument("out")
    p.add_argument("--seed-think-rows", action="store_true")
    a = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.dump_dir, "rank*.npz"))):
        b = np.load(path)
        for i in range(b["input_ids"].shape[0]):
            doc = b["document_ids"][i]
            ids = b["input_ids"][i]
            lm = b["loss_masks"][i] > 0
            valid = doc >= 0
            change = np.flatnonzero(np.diff(doc[valid])) + 1
            lens = np.diff(np.concatenate([[0], change, [int(valid.sum())]]))
            rows.append({"src": os.path.basename(path), "ids": ids, "lm": lm, "doc_lens": lens})
    ntok = sum(int(r["lm"].sum()) for r in rows)
    cross = 0
    for r in rows:
        ends = np.cumsum(r["doc_lens"]) - 1  # last position of each document predicts across
        cross += int(r["lm"][ends].sum())
    print("ROWS", len(rows), "LOSS-TOKENS", ntok, "CROSS-BOUNDARY", cross, flush=True)

    import transformers

    cfg = transformers.AutoConfig.from_pretrained(a.hf_dir, trust_remote_code=True)
    if "Vision" in (cfg.architectures or [""])[0]:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            a.hf_dir, dtype=torch.bfloat16, trust_remote_code=True
        )
        lm = model.language_model
    else:
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            a.hf_dir, dtype=torch.bfloat16, trust_remote_code=True
        )
    lm = lm.cuda().eval()
    emb, head = lm.model.embed_tokens.weight, lm.lm_head.weight
    if a.seed_think_rows:
        with torch.no_grad():
            for w in ([emb] if head is emb else [emb, head]):
                means = {t: w[s].float().mean(0) for t, s in THINK_ROWS}
                for t, m in means.items():
                    w[t] = m.to(w.dtype)
        print("SEEDED", THINK_ROWS, flush=True)

    tot, n_in, per_row, t0 = 0.0, 0, [], time.time()
    with torch.no_grad():
        for ri, r in enumerate(rows):
            rs, rn, s0 = 0.0, 0, 0
            for L in r["doc_lens"].tolist():
                ids, lmask = r["ids"][s0 : s0 + L], r["lm"][s0 : s0 + L]
                s0 += L
                if L < 2 or not lmask[:-1].any():
                    continue
                x = torch.tensor(ids.astype(np.int64), device="cuda")[None]
                h = lm.model(input_ids=x).last_hidden_state[0][:-1]
                tgt = torch.tensor(ids[1:].astype(np.int64), device="cuda")
                msk = torch.tensor(lmask[:-1], device="cuda")
                for s in range(0, h.shape[0], 4096):
                    lg = (h[s : s + 4096] @ head.T).float()
                    nll = torch.nn.functional.cross_entropy(lg, tgt[s : s + 4096], reduction="none")
                    rs += float((nll * msk[s : s + 4096]).sum())
                    rn += int(msk[s : s + 4096].sum())
            tot += rs
            n_in += rn
            per_row.append({"row": ri, "src": r["src"], "docs": len(r["doc_lens"]), "loss_tokens": rn, "ce": rs / max(rn, 1)})
            print("ROW", ri, r["src"], rn, round(rs / max(rn, 1), 6), f"{time.time() - t0:.0f}s", flush=True)
    res = {
        "loss_tokens": n_in,
        "cross_boundary_tokens": cross,
        "ce": tot / max(n_in, 1),
        "rows": per_row,
        "hf": a.hf_dir,
        "seeded": a.seed_think_rows,
        "batch": a.dump_dir,
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print("REFCE", json.dumps({k: res[k] for k in ("loss_tokens", "cross_boundary_tokens", "ce")}), flush=True)


if __name__ == "__main__":
    main()
