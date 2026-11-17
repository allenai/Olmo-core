"""FLOP-aware length sampling of a training pool (idea ``rows``, early_loss_proxy loop).

Writes a SUBSET of a converter shard (raw memmaps, rows end in EOS; per-task part files kept) in which each row is
kept independently with a probability that depends only on its token length:

* ``pow:L0:a``  keep prob = min(1, (len / L0) ** -a)   -> rows drawn ~ len^-a above L0 (no duplicates)
* ``cut:L:p``   keep prob = 1 if len <= L else p      -> short rows plus a random tail of long rows

The trainer (``--row-len-keep SPEC``) points ``--data`` at the subset and, unless ``--max-tokens`` is given, keeps
the LR schedule of the FULL pool (max_duration = full-pool rows x seq_len tokens, i.e. the same number of
optimizer steps, warmup and decay as the reference arms); the loader wraps into a second epoch only if a run
outlasts the subset. Sidecars (gold fingerprints, row_info) are content- or full-pool-keyed and are not copied.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np


def keep_prob(lens: np.ndarray, spec: str) -> np.ndarray:
    kind, x, y = spec.split(":")
    lens = lens.astype(np.float64)
    if kind == "pow":
        return np.minimum(1.0, (lens / float(x)) ** (-float(y)))
    if kind == "cut":
        return np.where(lens <= float(x), 1.0, float(y))
    raise ValueError(f"--row-len-keep: unknown spec {spec!r} (pow:L0:a | cut:L:p)")


def build(src: str, out_root: str, spec: str, seed: int = 0) -> str:
    """Build (once, atomically) the subset of ``src`` for ``spec``; returns its directory."""
    meta = json.load(open(os.path.join(src, "metadata.json")))
    key = hashlib.sha1(f"{os.path.abspath(src)}|{meta.get('num_tokens')}|{spec}|{seed}".encode()).hexdigest()[:12]
    out = os.path.join(out_root, f"rowpool_{key}")
    if os.path.exists(os.path.join(out, "metadata.json")):
        return out
    eos = int(meta["eos_token_id"])
    dtype, mdtype = np.dtype(meta.get("dtype", "uint32")), np.dtype(meta.get("mask_dtype", "bool"))
    tmp = f"{out}.tmp{os.environ.get('LOCAL_RANK', '0')}_{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(seed)
    stats, all_lens, n_loss = {}, [], 0
    for f in sorted(glob.glob(os.path.join(src, "token_ids_part_*.npy"))):
        ids = np.memmap(f, dtype=dtype, mode="r")
        lab = np.memmap(f.replace("token_ids_part_", "labels_mask_"), dtype=mdtype, mode="r")
        ends = np.nonzero(np.asarray(ids) == eos)[0]
        starts = np.concatenate([[0], ends[:-1] + 1])
        lens = ends - starts + 1
        keep = rng.random(len(lens)) < keep_prob(lens, spec)
        sel = np.nonzero(keep)[0]
        part = os.path.basename(f)
        np.concatenate([np.asarray(ids[starts[i]:ends[i] + 1]) for i in sel]).astype(dtype).tofile(
            os.path.join(tmp, part))
        lsel = np.concatenate([np.asarray(lab[starts[i]:ends[i] + 1]) for i in sel]).astype(mdtype)
        lsel.tofile(os.path.join(tmp, part.replace("token_ids_part_", "labels_mask_")))
        n_loss += int(lsel.sum())
        all_lens.append(lens[sel])
        bins = [0, 6144, 12288, 1 << 30]
        stats[part] = {"rows_in": int(len(lens)), "rows_kept": int(len(sel)),
                       "kept_by_len<=6k|<=12k|>12k": np.histogram(lens[sel], bins)[0].tolist(),
                       "in_by_len<=6k|<=12k|>12k": np.histogram(lens, bins)[0].tolist(),
                       "mean_len_in": float(lens.mean()), "mean_len_kept": float(lens[sel].mean())}
    lens = np.concatenate(all_lens)
    m = {k: v for k, v in meta.items() if k != "merged_from"}
    m.update(num_instances=int(len(lens)), num_tokens=int(lens.sum()), max_example_len=int(lens.max()),
             min_example_len=int(lens.min()), num_loss_tokens=n_loss,
             row_len_keep={"src": src, "spec": spec, "seed": seed, "src_num_instances": int(meta["num_instances"]),
                           "src_num_tokens": int(meta["num_tokens"]), "parts": stats})
    json.dump(m, open(os.path.join(tmp, "metadata.json"), "w"), indent=1)
    try:
        os.rename(tmp, out)  # another rank may have won the race with identical content
    except OSError:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return out
