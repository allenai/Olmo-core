"""
Full-context dev cross-entropy for the olmo3_cptdrop screen (records/olmo3-cptdrop-screen-plan.md),
one OLMo 3 7B checkpoint, one GPU.

Every checkpoint is scored the same way: teacher-forced CE over the ``labels_mask`` tokens of each
fixed-length row, dense attention exactly as pretrained (sliding windows kept on 24 layers, full
attention on 8), no dropping. Reported:

  overall        token-weighted mean CE over every scored target token
  bins           the same, by TARGET position in the row: 0-4k / 4-8k / 8-16k / 16-32k (bins past the
                 row length are omitted, so a 4k row has one bin = overall)
  per_row        per row and per bin: loss sum + token count, so paired deltas between two
                 checkpoints (``compare_devce.py``) get a per-row standard error
  repeat         CE by target type (grad_composition's ``repeat_category``, N=4, FAR=4096): novel (the 4-gram
                 ending at the target never occurred earlier in the row), near (its last earlier occurrence ends
                 <= 4096 back, inside every sliding window) and far (it occurred only > 4096 back). Near = local
                 copying; far = long-range use.
  ctx_gain       (``--ctx-window W``) long-context use, separated from domain adaptation: the last W/2 targets
                 of each row scored twice, with the full row as context and with only the last W tokens
                 (each target then sees W/2..W tokens back). gain = CE_window - CE_full on the same targets
                 (> 0: the model uses context beyond W); per-row paired SE.

``eval_size`` is the number of ROWS (each a 32k or 4k window); token counts are reported per bin.

The model is ``TransformerConfig.olmo3_7B`` (vocab 100352, flash_2) plus YaRN on the full-attention
layers only, with the parameters of olmo3_configs.py / the released stage-3 config:
``--yarn-factor none`` (stage2 as is, plain RoPE), ``4`` (the 32k CPT runs), ``8`` (stage3).
An unknown ``--family`` raises: this script never falls back to a Qwen (or any other) config.

    python eval_devce.py --ckpt /data/prasann/olmo3_cptdrop/base/stage3-step11921 --yarn-factor 8 \\
        --dev /data/prasann/olmo3_cptdrop/data/dev32k --out stage3_dev32k.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

#: olmo-core's padded dolma2 embedding size (100278 real ids -> 100352 rows), as written by
#: ``convert_checkpoint_from_hf.py --model-arch olmo3_7b --tokenizer dolma2``.
OLMO3_VOCAB_SIZE = 100352
OLMO3_REAL_VOCAB = 100278
OLMO3_TOKENIZER = "allenai/dolma2-tokenizer"
#: YaRN parameters other than the factor: the released stage-3 config.json (and olmo3_configs.py).
YARN_KW = dict(beta_fast=32, beta_slow=1, old_context_len=8192)
#: Target-position bin edges; a bin is kept only if it starts inside the row.
BIN_EDGES = (0, 4096, 8192, 16384, 32768)
#: Logit rows per fp32 CE chunk (4096 x 100352 x 4 B = 1.6 GB).
CE_CHUNK = 4096


def log(m: str) -> None:
    print(f"[olmo3-devce] {m}", flush=True)


def _olmo3_7b(vocab_size: int, attn_backend: str):
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    return TransformerConfig.olmo3_7B(vocab_size=vocab_size, attn_backend=AttentionBackendName(attn_backend))


#: family -> (factory(vocab_size, attn_backend), vocab_size, tokenizer the shards must be built with)
FAMILIES = {"olmo3_7b": (_olmo3_7b, OLMO3_VOCAB_SIZE, OLMO3_TOKENIZER)}


def parse_yarn(v: str) -> Optional[float]:
    if str(v).lower() in ("none", "0", "1", "1.0"):
        return None
    f = float(v)
    if f <= 1:
        raise argparse.ArgumentTypeError(f"--yarn-factor must be none or > 1, got {v}")
    return f


def build_cfg(family: str, yarn_factor: Optional[float], attn_backend: str = "flash_2"):
    """The model config for ``family`` with YaRN ``yarn_factor`` (None = plain RoPE) on its full-attention layers."""
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.rope import YaRNRoPEScalingConfig

    if family not in FAMILIES:
        raise SystemExit(f"unknown --family {family!r} (known: {sorted(FAMILIES)}); refusing to guess a config")
    fac, vocab, _ = FAMILIES[family]
    cfg = fac(vocab, attn_backend)
    if yarn_factor is not None:
        # full_attn_layers_only=True: override blocks for exactly the non-sliding layers (3, 7, ..., 31)
        cfg = cfg.with_rope_scaling(YaRNRoPEScalingConfig(factor=yarn_factor, **YARN_KW), full_attn_layers_only=True)
        if not cfg.block_overrides:
            raise SystemExit("with_rope_scaling emitted no block_overrides: no full-attention layer got YaRN")
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    return cfg


def _yarn_by_layer(model_cfg: dict) -> Dict[str, Optional[float]]:
    """{layer: YaRN factor} read off a saved TransformerConfig dict (the checkpoint's config.json)."""
    out = {}
    for k, blk in (model_cfg.get("block_overrides") or {}).items():
        sc = (((blk or {}).get("sequence_mixer") or {}).get("rope") or {}).get("scaling")
        out[str(k)] = float(sc["factor"]) if sc else None
    return out


#: Tokenizers with dolma2's exact ids. train_ctc_suite records the patched OLMo 3 marker tokenizer (dolma2 with the
#: unused <|extra_id_1|>/<|extra_id_2|> RENAMED, ids unchanged; olmo3_configs.OLMO3_MARKER_TOKENIZER) in config.json.
_OLMO3_SAME_IDS = ("allenai/dolma2-tokenizer", "allenai/dolma3-tokenizer")
_OLMO3_MARKER_COPIES = ("ctc_olmo3/tokenizer", "Olmo-3-1025-7B-docchunk")


def _same_ids_tokenizer(tok: str, family: str) -> bool:
    if family != "olmo3_7b":
        return False
    return tok in _OLMO3_SAME_IDS or any(tok.rstrip("/").endswith(c) for c in _OLMO3_MARKER_COPIES)


def check_ckpt_config(ckpt_dir: str, cfg, family: str) -> dict:
    """Cross-check the architecture against the checkpoint's own config.json when it has one.

    Shape fields must match (raise otherwise). The YaRN factor is reported, not enforced: stage2 is
    scored with YaRN x4 by design although it was saved without scaling.
    """
    path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(path):
        log(f"no config.json beside the checkpoint ({path}); shape check left to the distcp load")
        return {"ckpt_config": None}
    raw = json.load(open(path))
    m = raw.get("model", raw)
    want = {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "vocab_size": cfg.vocab_size}
    got = {k: m.get(k) for k in want}
    if got != want:
        raise SystemExit(f"{path}: architecture {got} != {family} config {want}")
    tok = ((raw.get("dataset") or {}).get("tokenizer") or {}).get("identifier")
    if tok is not None and tok != FAMILIES[family][2] and not _same_ids_tokenizer(tok, family):
        raise SystemExit(f"{path}: tokenizer {tok!r} != {FAMILIES[family][2]!r}")
    saved = _yarn_by_layer(m)
    built = {str(k): float(b.sequence_mixer.rope.scaling.factor) for k, b in (cfg.block_overrides or {}).items()
             if b.sequence_mixer.rope is not None and b.sequence_mixer.rope.scaling is not None}
    saved_f = sorted({v for v in saved.values() if v is not None})
    built_f = sorted(set(built.values()))
    if saved_f != built_f:
        log(f"NOTE: checkpoint was saved with YaRN {saved_f or 'none'}; scoring with YaRN {built_f or 'none'}")
    return {"ckpt_config": path, "ckpt_yarn_factors": saved_f, "eval_yarn_factors": built_f,
            "eval_yarn_layers": sorted(int(k) for k in built)}


def find_distcp(root: str) -> str:
    if os.path.isfile(os.path.join(root, ".metadata")):
        return root
    cands = sorted(glob.glob(f"{root}/model_and_optim") + glob.glob(f"{root}/step*/model_and_optim"),
                   key=lambda p: (len(p), p))
    if not cands:
        raise SystemExit(f"no model_and_optim (distcp) under {root}")
    return cands[-1]


def load_shard(shard: str, max_rows: Optional[int], family: str, row_start: int = 0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(rows, L) int64 tokens + (rows, L) bool target masks from a train_ctc_suite fixed-length shard.

    The ``.npy`` files are RAW (no header): read with memmap at offset 0, never ``np.load``.
    """
    meta = json.load(open(f"{shard}/metadata.json"))
    tok = meta.get("tokenizer")
    if tok != FAMILIES[family][2]:
        raise SystemExit(f"{shard}: built with tokenizer {tok!r}, family {family} needs {FAMILIES[family][2]!r}")
    L = int(meta["max_example_len"])
    if int(meta["min_example_len"]) != L:
        raise SystemExit(f"{shard}: rows are not fixed-length ({meta['min_example_len']}..{L})")
    ids_f = sorted(glob.glob(f"{shard}/token_ids_part_*.npy"))
    msk_f = sorted(glob.glob(f"{shard}/labels_mask_*.npy"))
    if len(ids_f) != 1 or len(msk_f) != 1:
        raise SystemExit(f"{shard}: expected one token part + one mask part, got {ids_f} / {msk_f}")
    ids = np.memmap(ids_f[0], dtype=np.dtype(meta["dtype"]), mode="r")
    msk = np.memmap(msk_f[0], dtype=np.dtype(meta.get("mask_dtype", "bool")), mode="r")
    n = int(meta["num_instances"])
    if ids.size != n * L or msk.size != n * L:
        raise SystemExit(f"{shard}: {ids.size} tokens / {msk.size} mask entries != {n} rows x {L}")
    if row_start >= n:
        raise SystemExit(f"{shard}: --row-start {row_start} >= {n} rows")
    n = n - row_start
    if max_rows:
        n = min(n, max_rows)
    lo = row_start * L
    rows = np.asarray(ids[lo : lo + n * L], dtype=np.int64).reshape(n, L)
    masks = np.asarray(msk[lo : lo + n * L], dtype=bool).reshape(n, L)
    hi = int(rows.max())
    if hi >= OLMO3_REAL_VOCAB:
        raise SystemExit(f"{shard}: token id {hi} >= dolma2 vocab {OLMO3_REAL_VOCAB}")
    return rows, masks, meta


REPEAT_N, REPEAT_FAR = 4, 4096
REPEAT_NAMES = ("novel", "near", "far")


def repeat_category(ids) -> np.ndarray:
    """(L,) int8 per TARGET index j (same convention as ``row_token_ce``): 0 novel, 1 near repeat (the 4-gram
    ``ids[j-3 : j+1]`` last occurred ending <= REPEAT_FAR positions earlier), 2 far-only (only earlier than that).
    Copied from research/token_routing/grad_composition/cpt_analyze.py (branch claude/grad-composition-6f09c7)."""
    L = len(ids)
    cat = np.zeros(L, dtype=np.int8)
    last = {}
    for j in range(REPEAT_N - 1, L):
        key = tuple(ids[j - REPEAT_N + 1 : j + 1])
        p = last.get(key)
        if p is not None:
            cat[j] = 1 if j - p <= REPEAT_FAR else 2
        last[key] = j
    return cat


def bins_for(L: int, edges: Sequence[int] = BIN_EDGES) -> List[Tuple[str, int, int]]:
    """[(name, lo, hi)] target-position bins clipped to a row of length ``L``."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo >= L:
            break
        hi = min(hi, L)
        out.append((f"{lo // 1024}k-{hi // 1024}k", lo, hi))
    return out


@torch.no_grad()
def row_token_ce(model, x: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    """(L,) fp32 per-TARGET-position CE for one row, NaN where unscored.

    Index ``j`` holds the loss of predicting token ``j`` from tokens ``< j`` (so index 0 is always
    NaN), scored iff ``tgt_mask[j]``. Logits come from the model in its compute dtype; CE is fp32,
    in chunks so the fp32 logits never exist at full length.
    """
    logits = model(x[None])[0]  # (L, V)
    L = x.shape[0]
    out = torch.full((L,), float("nan"), device=x.device, dtype=torch.float32)
    for s in range(0, L - 1, CE_CHUNK):
        e = min(s + CE_CHUNK, L - 1)
        ce = F.cross_entropy(logits[s:e].float(), x[s + 1 : e + 1], reduction="none")
        out[s + 1 : e + 1] = ce
    del logits
    out[~tgt_mask] = float("nan")
    return out


def score_rows(model, rows: np.ndarray, masks: np.ndarray, device, log_every=(1, 2, 5, 10),
               ctx_window: int = 0) -> dict:
    """Per-row, per-bin loss sums and counts (float64 / int64 numpy) for every row; with ``ctx_window`` W also
    per-row (full-context sum, W-window sum, count) over the last W/2 targets (see ``ctx_gain``)."""
    n, L = rows.shape
    bins = bins_for(L)
    sums = np.zeros((n, len(bins)), dtype=np.float64)
    cnts = np.zeros((n, len(bins)), dtype=np.int64)
    ctx = np.zeros((n, 3), dtype=np.float64)
    rep_s = np.zeros((n, 3), dtype=np.float64)
    rep_c = np.zeros((n, 3), dtype=np.int64)
    if ctx_window and not (0 < ctx_window < L and ctx_window % 2 == 0):
        raise SystemExit(f"--ctx-window {ctx_window} must be even and < row length {L}")
    t0 = time.time()
    for i in range(n):
        x = torch.from_numpy(rows[i]).to(device)
        m = torch.from_numpy(masks[i]).to(device)
        ce = row_token_ce(model, x, m).cpu().numpy().astype(np.float64)
        cat = repeat_category(rows[i].tolist())
        okr = ~np.isnan(ce)
        for k in range(3):
            sel = okr & (cat == k)
            rep_s[i, k] = ce[sel].sum()
            rep_c[i, k] = int(sel.sum())
        if ctx_window:
            W, T = ctx_window, ctx_window // 2
            cw = row_token_ce(model, x[L - W:], m[L - W:]).cpu().numpy().astype(np.float64)[W - T:]
            cf = ce[L - T:]
            ok = ~np.isnan(cf) & ~np.isnan(cw)
            ctx[i] = (cf[ok].sum(), cw[ok].sum(), int(ok.sum()))
        for b, (_, lo, hi) in enumerate(bins):
            v = ce[lo:hi]
            ok = ~np.isnan(v)
            sums[i, b] = v[ok].sum()
            cnts[i, b] = int(ok.sum())
        if (i + 1) in log_every or (i + 1) % 32 == 0 or i + 1 == n:
            el = time.time() - t0
            log(f"row {i + 1}/{n} running CE {sums[: i + 1].sum() / max(cnts[: i + 1].sum(), 1):.4f} "
                f"({el:.0f}s, {el / (i + 1):.2f}s/row, ETA {el / (i + 1) * (n - i - 1):.0f}s)")
    return {"bins": bins, "sums": sums, "cnts": cnts, "ctx": ctx if ctx_window else None, "rep_s": rep_s, "rep_c": rep_c,
            "score_seconds": time.time() - t0}


def summarize_ctx(ctx: np.ndarray, window: int) -> dict:
    """ctx_gain summary from per-row (full sum, window sum, count)."""
    full, win, c = ctx[:, 0], ctx[:, 1], ctx[:, 2]
    ok = c > 0
    d = (win[ok] - full[ok]) / c[ok]
    return {"window": window, "tail_targets_per_row": window // 2, "tokens": int(c.sum()), "rows": int(ok.sum()),
            "ce_full_tail": float(full.sum() / c.sum()), "ce_window_tail": float(win.sum() / c.sum()),
            "gain": float((win.sum() - full.sum()) / c.sum()),
            "gain_row_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}


def summarize(sums: np.ndarray, cnts: np.ndarray, bins) -> dict:
    """Token-weighted CE overall + per bin, plus the row-mean CE and its SE across rows."""

    def one(s: np.ndarray, c: np.ndarray) -> dict:
        ok = c > 0
        row_ce = s[ok] / c[ok]
        k = int(ok.sum())
        return {"ce": float(s.sum() / c.sum()) if c.sum() else None,
                "tokens": int(c.sum()), "rows": k,
                "row_mean_ce": float(row_ce.mean()) if k else None,
                "row_se": float(row_ce.std(ddof=1) / math.sqrt(k)) if k > 1 else None}

    out = {"overall": one(sums.sum(1), cnts.sum(1))}
    out["bins"] = {name: one(sums[:, b], cnts[:, b]) for b, (name, _, _) in enumerate(bins)}
    return out


def summarize_repeat(rep_s: np.ndarray, rep_c: np.ndarray) -> dict:
    """CE by target type (novel / near / far), with the same fields as a position bin."""
    return {nm: summarize(rep_s[:, [k]], rep_c[:, [k]], [(nm, 0, 1)])["overall"] for k, nm in enumerate(REPEAT_NAMES)}


#: Checkpoint parameters a dense dev-CE eval legitimately ignores: the soft-token slot projector (the eval runs the
#: model dense, without compaction or slots).
EVAL_UNUSED_PREFIXES = ("pooled_projector.",)


def _ckpt_model_keys(ckpt: str) -> set:
    """Model parameter names stored in the distcp checkpoint (``model.`` prefix stripped)."""
    from torch.distributed.checkpoint import FileSystemReader

    md = FileSystemReader(find_distcp(ckpt)).read_metadata()
    return {k[len("model."):] for k in md.state_dict_metadata if k.startswith("model.")}


def reinstall_early_lora(ckpt: str, model) -> Optional[dict]:
    """A low-rank-early-stack run (``--st-otr-early-lora``) stores ``lora_A`` / ``lora_B`` keys the base model lacks.
    Re-install the adapters before the load so the trained model (not the base) is scored. The block comes from
    config.json["early_lora"]; if that is missing (router branch <= 1e5a14c85 never exported it: SoftToken defines
    export_config twice) but the checkpoint holds lora keys, it is rebuilt from the keys (n_blocks) and
    provenance.json (rank, scale). No adapters -> no-op; the import is lazy (older trees lack the module)."""
    path = os.path.join(ckpt, "config.json")
    block = json.load(open(path)).get("early_lora") if os.path.isfile(path) else None
    if not block:
        try:
            lora = [k for k in _ckpt_model_keys(ckpt) if k.endswith(".lora_A")]
        except (Exception, SystemExit):
            lora = []
        if not lora:
            return None
        prov_p = os.path.join(ckpt, "provenance.json")
        prov = json.load(open(prov_p)) if os.path.isfile(prov_p) else {}
        flat = json.dumps(prov)
        import re as _re

        r = _re.search(r'"st_otr_early_lora": *(\d+)', flat)
        sc = _re.search(r'"st_otr_early_lora_scale": *([0-9.eE+-]+)', flat)
        if r is None or sc is None:
            raise SystemExit(f"{ckpt}: checkpoint has {len(lora)} lora_A keys but neither config.json nor "
                             "provenance.json records the adapter rank / scale; refusing to score it as the base")
        n_blocks = 1 + max(int(k.split(".")[1]) for k in lora if k.startswith("blocks."))
        block = {"n_blocks": n_blocks, "rank": int(r.group(1)), "scale": float(sc.group(1))}
        log(f"early_lora block missing from config.json; rebuilt from checkpoint keys + provenance: {block}")
    from olmo_core.nn.early_lora import enable_from_config_block

    enable_from_config_block(model, block)
    return block


def load_model(ckpt: str, cfg, device: str, max_len: int):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    model = cfg.build(init_device="cpu")
    reinstall_early_lora(ckpt, model)
    # never score a checkpoint whose parameters the built model would silently skip (distcp loads only the
    # keys the model asks for): 2026-10-24, LoRA runs scored without their adapters
    want = set(model.state_dict().keys())
    try:
        have = _ckpt_model_keys(ckpt)
    except (Exception, SystemExit):
        have = None
    if have is not None:
        # soft-token training adds a slot projector the dense eval never uses (slot-less / dense scoring)
        extra = sorted(k for k in have - want if not k.startswith(EVAL_UNUSED_PREFIXES))
        if extra:
            raise SystemExit(f"{ckpt}: checkpoint holds {len(extra)} model keys the eval model lacks "
                             f"(e.g. {extra[:3]}); refusing to score a partial model")
    t0 = time.time()
    load_model_and_optim_state(find_distcp(ckpt), model)
    load_s = time.time() - t0
    model = model.to(device=device, dtype=torch.bfloat16 if device != "cpu" else torch.float32).eval()
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_len + 8, torch.device(device))
    return model, load_s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (holds model_and_optim/) or the distcp dir itself")
    ap.add_argument("--dev", required=True, help="fixed-length shard dir (dev32k / short4k)")
    ap.add_argument("--yarn-factor", type=parse_yarn, required=True,
                    help="none (stage2 as is) | 4 (32k CPT runs) | 8 (stage3); full-attention layers only")
    ap.add_argument("--family", default="olmo3_7b")
    ap.add_argument("--rows", type=int, default=0, help="first N rows only (0 = all)")
    ap.add_argument("--row-start", type=int, default=0,
                    help="skip the first K rows (row chunks across GPUs; merge_devce.py joins the chunk JSONs)")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--ctx-window", type=int, default=0,
                    help="also score the last W/2 targets with only the last W tokens as context (ctx_gain); 0 = off")
    ap.add_argument("--label", default="", help="free-form name stored in the JSON (e.g. stage2-yarn4)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t_all = time.time()
    rows, masks, meta = load_shard(a.dev, a.rows or None, a.family, a.row_start)
    n, L = rows.shape
    log(f"{n} rows x {L} tokens, {int(masks[:, 1:].sum())} scored targets, from {a.dev}")
    cfg = build_cfg(a.family, a.yarn_factor, a.attn_backend)
    ck_info = check_ckpt_config(a.ckpt, cfg, a.family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, load_s = load_model(a.ckpt, cfg, device, L)
    log(f"loaded {find_distcp(a.ckpt)} in {load_s:.0f}s (yarn={a.yarn_factor}, {a.attn_backend}, {device})")
    sc = score_rows(model, rows, masks, device, ctx_window=a.ctx_window)
    summ = summarize(sc["sums"], sc["cnts"], sc["bins"])
    summ["repeat"] = summarize_repeat(sc["rep_s"], sc["rep_c"])
    if a.ctx_window:
        summ["ctx_gain"] = summarize_ctx(sc["ctx"], a.ctx_window)
    out = {
        "label": a.label, "family": a.family, "ckpt": a.ckpt, "dev": a.dev, "yarn_factor": a.yarn_factor,
        "attn_backend": a.attn_backend, "eval_size": n, "eval_size_unit": "rows", "row_len": L,
        "cot": "no-cot", "metric": "teacher-forced CE (nats/token) over labels_mask targets, dense attention",
        "summary": summ,
        "bins": [{"name": nm, "lo": lo, "hi": hi} for nm, lo, hi in sc["bins"]],
        "per_row": {"loss_sum": sc["sums"].tolist(), "tokens": sc["cnts"].tolist(),
                    "repeat_loss_sum": sc["rep_s"].tolist(), "repeat_tokens": sc["rep_c"].tolist(),
                    **({"ctx_full_win_cnt": sc["ctx"].tolist()} if a.ctx_window else {})},
        "ctx_window": a.ctx_window,
        "row_ids": list(range(a.row_start, a.row_start + n)),
        "shard_meta": {k: meta.get(k) for k in ("source", "buckets", "tokenizer", "num_instances", "seed")},
        **ck_info,
        "timing_s": {"load": load_s, "score": sc["score_seconds"], "total": time.time() - t_all},
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        # the staged node copy is a `git archive` (no .git): the wrapper passes the commit in
        "git_commit": os.environ.get("EVAL_GIT_COMMIT") or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(),
        "argv": sys.argv if argv is None else argv,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    o = summ["overall"]
    log(f"SUMMARY {a.label or a.ckpt}: overall CE {o['ce']:.4f} ({o['tokens']} tokens, eval_size={n} rows) | "
        + " ".join(f"{k} {v['ce']:.4f}" for k, v in summ["bins"].items())
        + " | " + " ".join(f"{k} {v['ce']:.4f}" for k, v in summ["repeat"].items() if v["ce"] is not None)
        + (f" | ctx_gain@{a.ctx_window} {summ['ctx_gain']['gain']:+.4f}" if a.ctx_window else ""))
    log(f"timing: load {load_s:.0f}s, score {sc['score_seconds']:.0f}s; wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
