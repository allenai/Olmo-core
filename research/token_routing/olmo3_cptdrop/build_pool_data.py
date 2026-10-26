"""
Build the olmo3_cptdrop CPT shards from the Dolma 3 Longmino pool (records/olmo3-cptdrop-screen-plan.md).

Three outputs, all in the ``train_ctc_suite.py`` shard layout (raw uint32 ``token_ids_part_0000.npy``,
raw bool ``labels_mask_0000.npy``, ``metadata.json``) plus a ``rows.tsv`` provenance sidecar:

* ``train32k`` -- rows of ``row_len`` tokens = EOS + one contiguous window of ONE document. Documents
  come from the pool's 2^15 / 2^16 length buckets (nominally 32k-128k), at most ``max_windows``
  windows per document, documents shorter than one window skipped. No marker tokens: the token-drop
  arms define 512-token blocks by position. Rows are globally shuffled, so every budget prefix mixes
  topics.
* ``dev32k`` -- the first window of documents from the 2^17+ buckets (nominally >=128k), which the
  OLMo 3 7B stage-3 mix never sampled. Round-robin over topics. Disjoint from train by document id
  (asserted) and by bucket.
* ``short4k`` -- forgetting guard: 4,096-token rows packed from short (<4k) documents of one
  already-tokenized OLMo 3 stage-3 shard (dolma3_longmino_0625, mostly the short midtraining data).

The ``synth_cwe`` / ``synth_rex`` (synthetic task) folders are excluded. Tokenizer: allenai/dolma2-tokenizer
(identical to the dolma3 tokenizer), EOS 100257, no special tokens added.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import time
import urllib.request

import numpy as np

POOL = "allenai/dolma3_longmino_pool"
EOS = 100257
SHORT_SHARD_URL = (
    "https://olmo-data.org/preprocessed/dolma3_longmino_0625/allenai/dolma3-tokenizer/000500.npy"
)
DIR_RE = re.compile(r"^data/olmocr_science_pdfs-(?P<topic>.+)-2e(?P<b>\d+)/(?P<f>[^/]+\.jsonl\.zst)$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_pool(api):
    by = {}
    for p in api.list_repo_files(POOL, repo_type="dataset"):
        m = DIR_RE.match(p)
        if not m or m["topic"].startswith("synth_"):
            continue
        by.setdefault((m["topic"], int(m["b"])), []).append(p)
    return by


def read_docs(path):
    import pyarrow as pa

    bad = 0
    with pa.input_stream(path, compression="zstd") as s:
        for line in io.TextIOWrapper(s, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # a few pool lines are truncated; skip, never guess
                bad += 1
                continue
            yield d["id"], d["text"]
    if bad:
        log(f"skipped {bad} malformed JSON line(s) in {os.path.basename(path)}")


def windows(ids, row_len, max_windows):
    w = row_len - 1  # leading EOS
    out = []
    for k in range(min(max_windows, len(ids) // w)):
        out.append(ids[k * w : (k + 1) * w])
    return out


def write_shard(out, rows, prov, row_len, extra_meta):
    os.makedirs(out, exist_ok=True)
    tok = np.empty((len(rows), row_len), dtype=np.uint32)
    for i, r in enumerate(rows):
        tok[i] = r
    msk = np.ones_like(tok, dtype=bool)
    msk[:, 0] = False  # the leading EOS is not a target
    tok.reshape(-1).tofile(os.path.join(out, "token_ids_part_0000.npy"))
    msk.reshape(-1).tofile(os.path.join(out, "labels_mask_0000.npy"))
    with open(os.path.join(out, "rows.tsv"), "w") as f:
        f.write("row\tdoc_id\ttopic\tbucket\twindow\tdoc_tokens\n")
        for i, p in enumerate(prov):
            f.write(f"{i}\t" + "\t".join(str(x) for x in p) + "\n")
    meta = {
        "task": "cpt", "emit": "dense", "cot_mode": "none", "chunk_by": "positional512",
        "wrap_docs": False, "marker_set": "olmo3", "eos_token_id": EOS,
        "doc_start_id": None, "doc_end_id": None, "landmark_token_id": None, "pad_token_id": None,
        "mem_freq": None, "dtype": "uint32", "mask_dtype": "bool", "num_instances": len(rows),
        "num_dropped": 0, "num_tokens": int(tok.size), "num_loss_tokens": int(msk.sum()),
        "max_example_len": row_len, "min_example_len": row_len, "query_position": "n/a",
        "tokenizer": "allenai/dolma2-tokenizer", **extra_meta,
    }
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=1)
    log(f"wrote {out}: {len(rows)} rows x {row_len} = {tok.size:,} tokens")


def build_long(api, tok, files, n_rows, row_len, max_windows, cache, label, skip_ids=()):
    """Stream files in the given order, tokenize, cut windows until n_rows rows exist."""
    from huggingface_hub import hf_hub_download

    rows, prov, doc_lens, seen = [], [], [], set()
    t0, n_files = time.time(), 0
    for path in files:
        m = DIR_RE.match(path)
        local = hf_hub_download(POOL, path, repo_type="dataset", cache_dir=cache)
        batch = [(i, t) for i, t in read_docs(local) if i not in skip_ids]
        enc = tok([t for _, t in batch], add_special_tokens=False)["input_ids"]
        for (doc_id, _), ids in zip(batch, enc):
            assert doc_id not in seen, f"duplicate doc id {doc_id}"
            seen.add(doc_id)
            doc_lens.append(len(ids))
            for k, w in enumerate(windows(ids, row_len, max_windows)):
                rows.append(np.asarray([EOS] + w, dtype=np.uint32))
                prov.append((doc_id, m["topic"], m["b"], k, len(ids)))
        n_files += 1
        if n_files in (1, 2, 5, 10) or n_files % 10 == 0 or len(rows) >= n_rows:
            el = time.time() - t0
            eta = el / max(len(rows), 1) * max(n_rows - len(rows), 0)
            log(f"[{label}] files {n_files} docs {len(doc_lens)} rows {len(rows)}/{n_rows} "
                f"({el:.0f}s, eta {eta:.0f}s)")
        if len(rows) >= n_rows:
            break
    if len(rows) < n_rows:
        raise SystemExit(f"[{label}] only {len(rows)} rows from {n_files} files (wanted {n_rows})")
    return rows[:n_rows], prov[:n_rows], np.asarray(doc_lens), seen


def build_dev(api, tok, by, n_rows, row_len, cache, rng, train_ids):
    """First window of >=2^17-bucket documents, round-robin over topics."""
    from huggingface_hub import hf_hub_download

    topics = sorted({t for (t, b) in by if b >= 17})
    per_topic = -(-n_rows // len(topics))
    picked = {}
    for t in topics:
        files = [p for (tt, b), ps in by.items() if tt == t and b >= 17 for p in ps]
        rng.shuffle(files)
        got = []
        for path in files:
            m = DIR_RE.match(path)
            local = hf_hub_download(POOL, path, repo_type="dataset", cache_dir=cache)
            batch = [(i, x) for i, x in read_docs(local) if i not in train_ids]
            rng.shuffle(batch)
            for doc_id, text in batch:
                # only the first window is used; 12 chars/token is far above the ~4 of English text
                ids = tok(text[: row_len * 12], add_special_tokens=False)["input_ids"]
                w = windows(ids, row_len, 1)
                if w:
                    got.append((np.asarray([EOS] + w[0], dtype=np.uint32), (doc_id, t, m["b"], 0, len(ids))))
                if len(got) >= per_topic:
                    break
            if len(got) >= per_topic:
                break
        picked[t] = got
        log(f"[dev] {t}: {len(got)} rows")
    rows, prov = [], []
    for k in range(per_topic):  # round-robin so a prefix is still topic-balanced
        for t in topics:
            if k < len(picked[t]):
                rows.append(picked[t][k][0])
                prov.append(picked[t][k][1])
    assert not ({p[0] for p in prov} & train_ids), "dev/train document overlap"
    return rows[:n_rows], prov[:n_rows]


def build_short(n_rows, row_len, work, shards, seed):
    """Cut a shuffled stream of short (< row_len) documents drawn from several already-tokenized OLMo 3
    stage-3 shards into row_len windows (several documents per row). One shard alone is a single narrow
    source (000500 = LLM-written math solutions), hence the spread."""
    docs = []
    for sh in shards:
        url = SHORT_SHARD_URL.replace("000500", sh)
        dst = os.path.join(work, f"longmino0625_{sh}.npy")
        if not os.path.exists(dst):
            log(f"[short] downloading {url}")
            # olmo-data.org rejects urllib's default User-Agent with 403
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
            with urllib.request.urlopen(req) as r, open(dst + ".part", "wb") as f:
                while chunk := r.read(1 << 24):
                    f.write(chunk)
            os.replace(dst + ".part", dst)
        a = np.fromfile(dst, dtype=np.uint32)
        cuts = np.flatnonzero(a == EOS)
        starts = np.concatenate([[0], cuts[:-1] + 1])
        mine = [(sh, a[s : e + 1]) for s, e in zip(starts, cuts) if 64 <= e + 1 - s <= row_len - 1]
        log(f"[short] shard {sh}: {len(mine)} short docs")
        docs += mine
    random.Random(seed).shuffle(docs)
    w = row_len - 1
    rows, prov, buf, used = [], [], [], set()
    for sh, d in docs:  # each doc keeps its trailing EOS
        buf.append(d)
        used.add(sh)
        if sum(len(b) for b in buf) >= w:
            flat = np.concatenate(buf)
            rows.append(np.concatenate([[EOS], flat[:w]]).astype(np.uint32))
            prov.append((f"shards:{'+'.join(sorted(used))}", "short", 0, 0, w))
            buf, used = [flat[w:]] if len(flat) > w else [], set()
            if len(rows) >= n_rows:
                break
    log(f"[short] {len(docs)} short docs from {len(shards)} shards -> {len(rows)} rows")
    return rows, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--cache", required=True, help="HF download cache (node-local)")
    ap.add_argument("--row-len", type=int, default=32768)
    ap.add_argument("--train-rows", type=int, default=43000, help="43000 x 32768 = 1.41B tokens")
    ap.add_argument("--dev-rows", type=int, default=256)
    ap.add_argument("--short-rows", type=int, default=256)
    ap.add_argument("--short-row-len", type=int, default=4096)
    ap.add_argument("--max-windows", type=int, default=2)
    ap.add_argument("--train-buckets", default="15,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="train,dev,short")
    ap.add_argument("--train-name", default="train32k")
    ap.add_argument("--dev-name", default="dev32k")
    ap.add_argument("--short-name", default="short4k")
    ap.add_argument("--exclude-rows-tsv", default="",
                    help="comma list of rows.tsv files (e.g. the dev shards'): their doc ids never enter train "
                    "(needed when train comes from the 2^17+ buckets dev is drawn from, e.g. train128k)")
    ap.add_argument("--short-shards", default="000500",
                    help="comma list of dolma3_longmino_0625 shard ids for the short-document guard")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", cache_dir=a.cache)
    assert tok.eos_token_id == EOS, tok.eos_token_id
    api = HfApi()
    by = list_pool(api)
    log(f"pool: {len(by)} (topic, bucket) dirs, {sum(len(v) for v in by.values())} files")
    rng = random.Random(a.seed)
    only = set(a.only.split(","))
    train_ids: set = set()

    if "train" in only:
        buckets = {int(x) for x in a.train_buckets.split(",")}
        files = sorted(p for (t, b), ps in by.items() if b in buckets for p in ps)
        rng.shuffle(files)  # uniform over files = proportional to each (topic, bucket)'s size
        skip = set()
        for f in [x for x in a.exclude_rows_tsv.split(",") if x]:
            skip |= {ln.split("\t")[1] for ln in open(f).read().splitlines()[1:] if ln.strip()}
        if skip:
            log(f"[train] excluding {len(skip)} doc ids from {a.exclude_rows_tsv}")
        rows, prov, lens, train_ids = build_long(
            api, tok, files, a.train_rows, a.row_len, a.max_windows, a.cache, "train", skip_ids=skip
        )
        assert not (train_ids & skip), "excluded doc ids leaked into train"
        order = list(range(len(rows)))
        random.Random(a.seed + 1).shuffle(order)
        q = np.percentile(lens, [10, 50, 90, 99]).astype(int).tolist()
        write_shard(
            os.path.join(a.out_root, a.train_name), [rows[i] for i in order], [prov[i] for i in order],
            a.row_len,
            {"source": POOL, "buckets": sorted(buckets), "max_windows_per_doc": a.max_windows,
             "seed": a.seed, "docs_seen": int(len(lens)), "docs_used": len({p[0] for p in prov}),
             "doc_tokens_p10_p50_p90_p99": q, "doc_tokens_max": int(lens.max())},
        )
        with open(os.path.join(a.out_root, a.train_name, "doc_ids.txt"), "w") as f:
            f.write("\n".join(sorted(train_ids)))
    elif os.path.exists(os.path.join(a.out_root, a.train_name, "doc_ids.txt")):
        train_ids = set(open(os.path.join(a.out_root, a.train_name, "doc_ids.txt")).read().split())

    if "dev" in only:
        rows, prov = build_dev(api, tok, by, a.dev_rows, a.row_len, a.cache, rng, train_ids)
        write_shard(os.path.join(a.out_root, a.dev_name), rows, prov, a.row_len,
                    {"source": POOL, "buckets": ">=17", "seed": a.seed})

    if "short" in only:
        shards = a.short_shards.split(",")
        rows, prov = build_short(a.short_rows, a.short_row_len, a.cache, shards, a.seed)
        write_shard(os.path.join(a.out_root, a.short_name), rows, prov, a.short_row_len,
                    {"source": SHORT_SHARD_URL.replace("000500", "{" + ",".join(shards) + "}"),
                     "note": "short docs (< row_len) shuffled across shards, EOS-separated, packed"})
    log("DONE")


if __name__ == "__main__":
    main()
