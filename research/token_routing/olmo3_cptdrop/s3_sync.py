"""
Mirror local dirs to/from the olmo3_cptdrop S3 hub (s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/).
boto3 only (no aws CLI on the Berkeley nodes).

A file is skipped only when it is provably the same, never on size alone (a rebuilt shard has the same size:
that is how a stale c8k survived a "re-upload" on 2026-10-22):
  up    skip iff remote size == local size AND remote metadata src-mtime == local mtime (set by this script)
  down  skip iff local size == remote size AND the remote ETag equals the one recorded at the last download
        (.s3_etags.json in the local dir)
--force re-transfers everything.

    python s3_sync.py up   /data/prasann/olmo3_cptdrop/data/train32k data/train32k
    python s3_sync.py down /data/prasann/olmo3_cptdrop/data/train32k data/train32k
"""
from __future__ import annotations

import json
import os
import sys
import time

import boto3
from boto3.s3.transfer import TransferConfig

BUCKET = "ai2-llm"
ROOT = os.environ.get("S3_ROOT", "checkpoints/prasanns/_transfer/olmo3_cptdrop")  # S3_ROOT: another hub (e.g. _transfer/q3recipe)
CFG = TransferConfig(multipart_chunksize=128 << 20, max_concurrency=16)
ETAGS = ".s3_etags.json"


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    direction, local, rel = args[0], args[1].rstrip("/"), args[2].strip("/")
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "S3")).client("s3")
    prefix = f"{ROOT}/{rel}/"
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            remote[o["Key"][len(prefix):]] = (o["Size"], o["ETag"])
    t0, n, nbytes, skipped = time.time(), 0, 0, 0
    if direction == "up":
        for dp, _, fs in os.walk(local):
            for f in sorted(fs):
                if f == ETAGS:
                    continue
                p = os.path.join(dp, f)
                k = os.path.relpath(p, local)
                size, mtime = os.path.getsize(p), str(int(os.path.getmtime(p)))
                if not force and k in remote and remote[k][0] == size:
                    meta = s3.head_object(Bucket=BUCKET, Key=prefix + k).get("Metadata", {})
                    if meta.get("src-mtime") == mtime:
                        skipped += 1
                        continue
                s3.upload_file(p, BUCKET, prefix + k, Config=CFG, ExtraArgs={"Metadata": {"src-mtime": mtime}})
                n, nbytes = n + 1, nbytes + size
                print(f"up {k} {size/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    elif direction == "down":
        os.makedirs(local, exist_ok=True)
        epath = os.path.join(local, ETAGS)
        seen = json.load(open(epath)) if os.path.exists(epath) else {}
        for k, (size, etag) in sorted(remote.items()):
            p = os.path.join(local, k)
            if not force and os.path.exists(p) and os.path.getsize(p) == size and seen.get(k) == etag:
                skipped += 1
                continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            s3.download_file(BUCKET, prefix + k, p, Config=CFG)
            seen[k] = etag
            json.dump(seen, open(epath, "w"), indent=1)
            n, nbytes = n + 1, nbytes + size
            print(f"down {k} {size/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    else:
        raise SystemExit("direction must be up or down")
    print(f"DONE {direction} s3://{BUCKET}/{prefix}: {n} files, {nbytes/1e9:.2f} GB, {skipped} skipped, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
