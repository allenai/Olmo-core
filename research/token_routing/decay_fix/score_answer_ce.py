"""
Teacher-forced held-out ANSWER cross-entropy for fast8k mt4 checkpoints (tokdrop_lr large-data decay diagnosis,
2026-10-30).

Why: the 8-cell generative f1 mean of dense saturates near 0.93 from 3,904 rows/task on, so FLOPs-to-match read off
f1 turns a -0.005 deficit (SE of the 8-cell mean ~0.004) into a halving of the speedup. Answer CE on the same held-out
rows is continuous and does not saturate the same way, so it separates "the eval is saturated" from "the method
falls behind".

What is scored is exactly what the ladder generates from: the same rung files (the ds64 ladder files the fast8k
evals use), the same 500-row subsample (``load_unified_examples`` seeds random.sample(42)), the same prompt
rendering (``segment_prompt_to_chunks``, query position ``after``, no-cot, dense box-marker emitter), the same model
build (TransformerGenerationModuleConfig + post_build_hook_from_config, so LoRA adapters are re-installed from
config.json), and the same ``enable_document_chunk_attention(mode="chunked")`` call. The only difference: the
reference answer is appended (``include_answer=True``) and each answer token's CE is read off one forward.

Per row: ``ans`` = mean CE over the answer's content tokens (mask positions minus the chat scaffolding: <think>,
</think>, <|im_end|> and whitespace-only tokens); ``span`` = mean CE over every supervised position. Reported per
(task, rung): mean over rows of the per-row mean (``ans_row_mean``) and the token-weighted mean (``ans_tok_mean``),
plus per-row (sum, count) so two checkpoints can be compared with a paired SE.

    python research/token_routing/decay_fix/score_answer_ce.py --ckpt <export dir> --out /results/ce.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "recipes", "eval", "legacy_native"))
sys.path.insert(0, os.path.join(REPO, "src"))

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
#: research/token_routing/ds64_fast8k/launch_fast8k.py RUNGS (the files every fast8k mt4 eval scored)
RUNGS = {
    "contradiction": {"8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                      "2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"},
    "nq": {"8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
           "2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"},
    "oolong": {"8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
               "2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"},
    "outlier": {"8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                "2k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
}
#: eval_lc_native_docchunk_ladder.build_task_spec: task -> (loadtask, chunk_by, item_regex)
SPEC = {"contradiction": ("contradiction", "document", r"\|\|"), "nq": ("retrieval", "document", r"\|\|"),
        "oolong": ("oolong", "line", r"\|\|"), "outlier": ("outlier", "document", r"\|\|")}
SCAFFOLD = {"<think>", "</think>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"}


def log(m):
    print(f"[answer-ce] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="export dir (config.json + model_and_optim/); repeat to score several in one job")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,oolong,outlier")
    ap.add_argument("--rungs", default="8k,2k")
    ap.add_argument("--max-test", type=int, default=500)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--query-position", default="after")
    ap.add_argument("--max-len", type=int, default=20000)
    ap.add_argument("--weka-out", default="", help="also copy the JSON here (a weka path) as each ckpt finishes")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import ctc_eval.eval.eval_lc_native_docchunk_ladder as L
    from ctc_eval.eval.evaluate import load_unified_examples
    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import emit_document_chunk_dense, segment_prompt_to_chunks
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from olmo_core.nn.nested_ffn_moe import post_build_hook_from_config

    tok = L._tokenizer_with_retry(a.tokenizer)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    out = {"_meta": {"tokenizer": a.tokenizer, "query_position": a.query_position, "cot": "no-cot",
                     "max_test": a.max_test, "rungs": RUNGS}}
    if os.path.exists(a.out):
        out.update(json.load(open(a.out)))

    for ckpt in a.ckpt:
        name = os.path.basename(ckpt.rstrip("/"))
        if name in out and out[name].get("_done"):
            log(f"skip {name}: done")
            continue
        ns = argparse.Namespace(family="auto", model_path=ckpt, tokenizer=a.tokenizer)
        L.resolve_reserved_ids(ns, tok)
        pad_id = tok.pad_token_id if tok.pad_token_id not in (None, L.EOS_TOKEN_ID) else L.IM_END_ID
        gen_cfg = GenerationConfig(eos_token_id=L.EOS_TOKEN_ID, pad_token_id=pad_id, max_length=a.max_len,
                                   use_cache=True)
        t0 = time.time()
        gm = TransformerGenerationModuleConfig(gen_cfg, float8_config=None, dtype=DType("bfloat16"),
                                               compile_model=False).build(
            checkpoint_dir=ckpt, device=device, post_build_hook=post_build_hook_from_config(ckpt))
        model = gm.model
        model.eval()
        model.enable_document_chunk_attention(doc_start_id=L.DOC_START_ID, doc_end_id=L.DOC_END_ID,
                                              eos_id=L.EOS_TOKEN_ID, mode="chunked", pad_id=None)
        n_lora = len(getattr(model, "_early_lora", {}).get("modules", [])) if getattr(model, "_early_lora", None) else 0
        log(f"{name}: built in {time.time() - t0:.0f}s (lora linears: {n_lora})")
        try:
            from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune
            freeze_fla_length_autotune()
        except Exception as exc:  # noqa: BLE001
            log(f"fla autotune freeze unavailable: {exc}")
        res = {"_lora_linears": n_lora}
        for task in a.tasks.split(","):
            loadtask, chunk_by, item_regex = SPEC[task]
            for rung in a.rungs.split(","):
                path = RUNGS[task][rung]
                exs = load_unified_examples(path, a.max_test, task=loadtask, query_position=a.query_position,
                                            use_alpaca=True)
                rows, t1 = [], time.time()
                for gi, e in enumerate(exs):
                    raw = e.get("ex", e)
                    segs, _ids, _m = segment_prompt_to_chunks(
                        tok, raw, loadtask, query_position=a.query_position, cot_mode="none", chunk_by=chunk_by,
                        item_regex=item_regex, include_answer=True, doc_start_id=L.DOC_START_ID,
                        doc_end_id=L.DOC_END_ID)
                    ids, mask = emit_document_chunk_dense(segs)
                    sup = [i for i, m in enumerate(mask) if m]
                    if not sup or len(ids) > a.max_len:
                        rows.append(None)
                        continue
                    first = sup[0]
                    keep = len(ids) - first + 1  # logits at [first-1, len-1) predict tokens [first, len)
                    x = torch.tensor([ids], device=device)
                    with torch.inference_mode():
                        logits = model(x, logits_to_keep=keep)
                        if not torch.is_tensor(logits):
                            logits = logits.logits
                        lg = logits[0, :-1].float()  # predicts ids[first : len]
                        tgt = torch.tensor(ids[first:], device=device)
                        ce = F.cross_entropy(lg, tgt, reduction="none")
                        top1 = (lg.argmax(-1) == tgt)
                    ce, top1 = ce.tolist(), top1.tolist()
                    span = [j for j in range(len(tgt)) if mask[first + j]]
                    content = [j for j in span
                               if tok.convert_ids_to_tokens(ids[first + j]) not in SCAFFOLD
                               and tok.decode([ids[first + j]]).strip() != ""]
                    rows.append({"i": gi, "len": len(ids),
                                 "ans_sum": sum(ce[j] for j in content), "ans_n": len(content),
                                 "ans_top1": sum(top1[j] for j in content),
                                 "span_sum": sum(ce[j] for j in span), "span_n": len(span)})
                    if gi in (0, 1, 4, 9, 49) or gi % 100 == 99:
                        el = time.time() - t1
                        log(f"{name} {task}@{rung} row {gi + 1}/{len(exs)} {el:.0f}s "
                            f"ETA {el / (gi + 1) * (len(exs) - gi - 1):.0f}s")
                ok = [r for r in rows if r and r["ans_n"] > 0]
                n_tok = sum(r["ans_n"] for r in ok)
                cell = {"eval_size": len(exs), "scored": len(ok),
                        "ans_row_mean": sum(r["ans_sum"] / r["ans_n"] for r in ok) / max(1, len(ok)),
                        "ans_tok_mean": sum(r["ans_sum"] for r in ok) / max(1, n_tok),
                        "ans_top1": sum(r["ans_top1"] for r in ok) / max(1, n_tok),
                        "span_row_mean": sum(r["span_sum"] / r["span_n"] for r in ok) / max(1, len(ok)),
                        "rows": rows}
                res[f"{task}_{rung}"] = cell
                log(f"{name} {task}@{rung}: ans_row_mean={cell['ans_row_mean']:.4f} "
                    f"ans_tok_mean={cell['ans_tok_mean']:.4f} top1={cell['ans_top1']:.4f} "
                    f"(eval_size={len(exs)}, scored={len(ok)}, {time.time() - t1:.0f}s)")
        res["_done"] = True
        out[name] = res
        json.dump(out, open(a.out + ".tmp", "w"))
        os.replace(a.out + ".tmp", a.out)
        if a.weka_out:
            os.makedirs(os.path.dirname(a.weka_out), exist_ok=True)
            json.dump(out, open(a.weka_out, "w"))
        del gm, model
        torch.cuda.empty_cache()
    log(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
