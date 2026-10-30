"""
Repair the degenerate embeddings of the reserved marker tokens used by the document-chunk / landmark /
summary-token data paths.

Qwen never trains its unused special-token rows: ``<|box_start|>`` and ``<|box_end|>`` -- the document
markers -- keep their shared initialization, so their embeddings are *bit-identical* (cosine
similarity 1.0000). The same holds for the landmark, pad and summary rows, which live past the real
vocab in the padded region of the embedding matrix.

The consequence is that the model cannot tell an "open document" marker from a "close document" one:
every marker is the same out-of-distribution vector. The roles built from the token *ids* are still
correct, so the attention mask is right -- but the model's *perception* of the document structure is
destroyed. Empirically a Qwen3-0.6B trains to CE ~0.004 on a 100-claim contradiction shard with no
markers, and plateaus at CE ~0.79 (chance-level f1) on the byte-identical shard *with* markers.

This script gives each marker a distinct, in-distribution embedding by **copying the row of a real,
trained token** and adding a small perturbation so the markers can still specialize. Marker ids are
never loss targets (the label mask covers only the answer span), so only the input embedding matters.

.. warning::
   **Fixing the cosine is not enough -- the norm matters just as much.** The original version of this
   script built each marker as ``mean(trained rows) + noise`` renormed to the norm of ``<|im_start|>``.
   That made the markers mutually distinguishable (cos ~0.01) but left them at norm **0.396** against a
   trained-token median of **1.415** -- roughly 1/3.6 of a real token, pointing in a meaningless
   "average token" direction. RMSNorm rescales every position to the same RMS, so such a row is
   amplified into a full-strength *noise* vector at each marker position. On the leak-free
   document-chunked shards (where a marker precedes every ``Claim N:`` label) training then flatlines
   at CE ~0.79 for **every** attention mask -- including plain causal -- i.e. an unrestricted model
   cannot even memorize the data. Deleting the markers, or giving them trained donor rows, restores
   learning (CE 0.058 vs 0.79 at 375 steps, single-variable). The "tied output head" rationale for the
   small norm does not apply: the ``lm_head`` is **not** tied to the embeddings here.
   See ``records/n100-chunked-marker-position-bug.md``.

.. note::
   Building the model requires that family's kernels to be importable -- Qwen3.5 is a GDN/attention
   hybrid and needs ``triton``. Run this as a gantry job on the OLMo-core image rather than on a
   laptop. The **audit** half needs none of that: see ``--audit-only``, which reads just the embedding
   matrix out of the checkpoint via ``load_keys`` and runs anywhere.

Usage::

    # audit any base, anywhere (no model construction, no triton)
    python recipes/data/fix_marker_embeddings.py --audit-only \\
        --base /path/to/base/model_and_optim --family qwen3_5 --marker-set summary,pad

    # repair (needs the training image for a hybrid family)
    python recipes/data/fix_marker_embeddings.py \\
        --base /path/to/base/model_and_optim --out /path/to/fixed \\
        --family qwen3_5 --model-size 4B --marker-set doc_start,doc_end,summary,pad
"""

import argparse
import json
from typing import Dict, List

import torch

from olmo_core.data import TokenizerConfig
from olmo_core.data.document_chunk_landmark import ReservedIds, reserved_ids
from olmo_core.distributed.checkpoint import load_keys
from olmo_core.nn.transformer import TransformerConfig

EMB_KEY = "model.embeddings.weight"

#: Donor tokens whose (trained) rows seed each marker.
#:
#: For the boundary/landmark/pad markers a **delimiter** is the right donor: the model already reads
#: it as "a boundary is here", which is exactly the marker's job. The summary token is different --
#: its job is to be a *summary slot*, not a boundary -- so it is seeded from the word ``"Summary"``,
#: which the model has seen as a heading preceding a condensed restatement. All are verified to be
#: single trained tokens at runtime; a donor that splits is a hard error, not a warning.
DONOR_TOKENS: Dict[str, str] = {
    "doc_start": "«",
    "doc_end": "»",
    "landmark": "§",
    "pad": "¶",
    "summary": "Summary",
}

#: Tokenizer to resolve donors against, per family. Qwen3.5's vocabulary (248,320) is a different
#: vocabulary from Qwen3's (151,936), so using the wrong one resolves donors to the wrong rows.
DEFAULT_TOKENIZERS: Dict[str, str] = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3_5": "Qwen/Qwen3.5-0.8B",
}

#: Model builders by (family, size). Only what is needed to load the checkpoint into.
MODEL_BUILDERS = {
    ("qwen3", "0.6B"): "qwen3_0_6B",
    ("qwen3", "1.7B"): "qwen3_1_7B",
    ("qwen3", "4B"): "qwen3_4B",
    ("qwen3", "8B"): "qwen3_8B",
    ("qwen3_5", "0.8B"): "qwen3_5_0_8B",
    ("qwen3_5", "4B"): "qwen3_5_4B",
}

#: Both gates. A marker must be distinguishable from its neighbours AND in-distribution in scale.
COS_MAX = 0.9
NORM_RATIO_MIN, NORM_RATIO_MAX = 0.5, 2.0


def _marker_ids(ids_set: ReservedIds, marker_set: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in marker_set:
        if name not in DONOR_TOKENS:
            raise SystemExit(f"unknown marker {name!r}; expected some of {sorted(DONOR_TOKENS)}")
        tid = getattr(ids_set, name)
        if tid is None or tid < 0:
            raise SystemExit(f"family has no id registered for marker {name!r}")
        out[name] = int(tid)
    return out


def audit(emb: torch.Tensor, ids_set: ReservedIds, markers: Dict[str, int]) -> dict:
    """Measure both gates on an embedding matrix. Pure measurement -- no mutation."""
    trained = emb[: ids_set.real_vocab_size].float()
    norms = trained.norm(dim=-1)
    median_norm = norms.median().item()

    report: dict = {
        "embedding_shape": list(emb.shape),
        "real_vocab_size": ids_set.real_vocab_size,
        "trained_row_median_norm": median_norm,
        "marker_ids": markers,
        "marker_norm_ratios": {},
        "pairwise_cosine": {},
    }
    for name, tid in markers.items():
        report["marker_norm_ratios"][name] = emb[tid].float().norm().item() / median_norm
    names = list(markers)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = markers[names[i]], markers[names[j]]
            cos = torch.nn.functional.cosine_similarity(
                emb[a].float()[None], emb[b].float()[None]
            ).item()
            report["pairwise_cosine"][f"{names[i]}|{names[j]}"] = cos
            report.setdefault("bit_identical", {})[f"{names[i]}|{names[j]}"] = bool(
                torch.equal(emb[a], emb[b])
            )

    report["cosine_gate_pass"] = all(abs(c) < COS_MAX for c in report["pairwise_cosine"].values())
    report["norm_gate_pass"] = all(
        NORM_RATIO_MIN < r < NORM_RATIO_MAX for r in report["marker_norm_ratios"].values()
    )
    report["audit_pass"] = report["cosine_gate_pass"] and report["norm_gate_pass"]
    return report


def print_audit(report: dict) -> None:
    print(f"embedding matrix: {tuple(report['embedding_shape'])}")
    print(
        f"trained-row median norm={report['trained_row_median_norm']:.4f} "
        "(markers must land near this, NOT far below it)"
    )
    for name, ratio in report["marker_norm_ratios"].items():
        flag = "ok" if NORM_RATIO_MIN < ratio < NORM_RATIO_MAX else "OUT OF DISTRIBUTION"
        print(f"  {name:10s} id={report['marker_ids'][name]:>7}  norm={ratio:.3f}x median  {flag}")
    for pair, cos in report["pairwise_cosine"].items():
        flag = "ok" if abs(cos) < COS_MAX else "TOO SIMILAR"
        ident = report.get("bit_identical", {}).get(pair)
        print(f"  cos({pair}) = {cos:+.4f}  {flag}" + ("  [BIT-IDENTICAL]" if ident else ""))
    verdict = "no repair needed" if report["audit_pass"] else "REPAIR REQUIRED"
    print(
        f"VERDICT: cosine_gate={'PASS' if report['cosine_gate_pass'] else 'FAIL'} "
        f"norm_gate={'PASS' if report['norm_gate_pass'] else 'FAIL'} -> {verdict}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument("--out", help="destination dir (a model_and_optim/ is written under it)")
    ap.add_argument("--family", default="qwen3", help="RESERVED_IDS family key")
    ap.add_argument("--model-size", default="0.6B")
    ap.add_argument(
        "--marker-set",
        default="doc_start,doc_end,landmark,pad",
        help="comma-separated markers to repair. Recorded in the audit JSON so a base repaired for "
        "one experiment is not mistaken for repaired for another. The summary-token path uses "
        "'doc_start,doc_end,summary,pad'.",
    )
    ap.add_argument("--tokenizer", help="override the donor tokenizer (defaults by --family)")
    ap.add_argument("--audit-only", action="store_true", help="measure and exit; needs no kernels")
    ap.add_argument("--audit-json", help="write the audit report here")
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.1,
        help="per-marker noise added to the donor row, as a fraction of the trained-row std, so the "
        "markers can specialize away from their donors during training",
    )
    args = ap.parse_args()

    ids_set = reserved_ids(args.family)
    markers = _marker_ids(ids_set, [m.strip() for m in args.marker_set.split(",") if m.strip()])
    print(f"family={args.family} markers={markers}")

    # ---- audit (no model construction, so this runs anywhere) ----
    emb_ro = next(iter(load_keys(args.base, [EMB_KEY]))).float()
    report = audit(emb_ro, ids_set, markers)
    report.update(family=args.family, checkpoint=args.base, marker_set=args.marker_set)
    print_audit(report)

    if args.audit_only:
        if args.audit_json:
            with open(args.audit_json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"wrote {args.audit_json}")
        return

    if not args.out:
        raise SystemExit("--out is required unless --audit-only")

    # ---- repair ----
    from transformers import AutoTokenizer

    from olmo_core.distributed.checkpoint import (
        load_model_and_optim_state,
        save_model_and_optim_state,
    )

    tokenizer_name = args.tokenizer or DEFAULT_TOKENIZERS.get(args.family)
    if tokenizer_name is None:
        raise SystemExit(f"no default tokenizer for family {args.family!r}; pass --tokenizer")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    builder_name = MODEL_BUILDERS.get((args.family, args.model_size))
    if builder_name is None:
        raise SystemExit(
            f"no model builder for family={args.family} size={args.model_size}; "
            f"known: {sorted(MODEL_BUILDERS)}"
        )
    tok_cfg = getattr(TokenizerConfig, args.family)()
    model = getattr(TransformerConfig, builder_name)(vocab_size=tok_cfg.padded_vocab_size()).build(
        init_device="cpu"
    )
    load_model_and_optim_state(args.base, model)

    emb = model.embeddings.weight.data
    # Jitter is scaled by the trained-row std so it perturbs without moving the row out of
    # distribution; the norm itself is checked by ``audit`` afterwards.
    std = emb[: ids_set.real_vocab_size].float().std()

    g = torch.Generator().manual_seed(args.seed)
    for name, tid in markers.items():
        donor_str = DONOR_TOKENS[name]
        donor_ids = tok.encode(donor_str, add_special_tokens=False)
        if (
            len(donor_ids) != 1
            or donor_ids[0] >= ids_set.real_vocab_size
            or donor_ids[0] == ids_set.eos
        ):
            raise SystemExit(
                f"donor {donor_str!r} for {name} is not a single trained token in "
                f"{tokenizer_name}: {donor_ids}. Pick a different donor rather than seeding from "
                "a multi-token sequence."
            )
        before = emb[tid].float().norm()
        vec = emb[donor_ids[0]].float() + torch.randn(emb.shape[1], generator=g) * (
            std * args.jitter
        )
        emb[tid] = vec.to(emb.dtype)
        print(
            f"  {tid} {name:10s} <- {donor_str!r} (id {donor_ids[0]})   "
            f"norm {before:.4f} -> {emb[tid].float().norm():.4f}"
        )

    after = audit(emb, ids_set, markers)
    print_audit(after)
    assert after["cosine_gate_pass"], (
        "repair left two markers indistinguishable -- pick different donors. "
        f"cosines: {after['pairwise_cosine']}"
    )
    assert after["norm_gate_pass"], (
        "a repaired marker is out of distribution in norm. This is the exact failure the donor-row "
        f"init exists to prevent. ratios: {after['marker_norm_ratios']}"
    )
    print("markers are distinguishable AND in-distribution in norm")

    save_model_and_optim_state(f"{args.out}/model_and_optim", model, save_overwrite=True)
    print(f"wrote fixed base -> {args.out}/model_and_optim")

    if args.audit_json:
        after.update(
            family=args.family, checkpoint=f"{args.out}/model_and_optim", marker_set=args.marker_set
        )
        with open(args.audit_json, "w") as f:
            json.dump({"before": report, "after": after}, f, indent=2)
        print(f"wrote {args.audit_json}")


if __name__ == "__main__":
    main()
