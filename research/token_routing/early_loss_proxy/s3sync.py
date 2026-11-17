"""Minimal boto3 directory/file sync for the Beaker early-loss-proxy path (no aws CLI on the login node or the image).

    python s3sync.py up   <local file|dir> s3://bucket/prefix   # uploads files whose size differs / are missing
    python s3sync.py down s3://bucket/prefix <local dir>        # downloads likewise (a prefix = a "directory")

Credentials: profile ``$AWS_PROFILE`` (default ``S3`` = real AWS, the profile beaker.md documents); falls back to
the default chain when that profile does not exist. Skips by size only (enough for write-once shards / result JSONs:
a result JSON rewritten with the same size is NOT re-copied -- pass ``--force`` to copy everything).
"""
from __future__ import annotations

import os
import sys


def client():
    import boto3
    import botocore.exceptions

    try:
        return boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "S3")).client("s3")
    except botocore.exceptions.ProfileNotFound:
        return boto3.Session().client("s3")


def split(url):
    assert url.startswith("s3://"), url
    b, _, k = url[5:].partition("/")
    return b, k.rstrip("/")


def listing(s3, bucket, prefix):
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            out[o["Key"][len(prefix) + 1:]] = o["Size"]
    return out


def main():
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    mode, src, dst = args
    s3 = client()
    n = nb = 0
    if mode == "up":
        bucket, prefix = split(dst)
        if os.path.isfile(src):
            files = {os.path.basename(src): src}
        else:
            files = {os.path.relpath(os.path.join(r, f), src): os.path.join(r, f)
                     for r, _, fs in os.walk(src) for f in fs}
        have = {} if force else listing(s3, bucket, prefix)
        for rel, p in sorted(files.items()):
            sz = os.path.getsize(p)
            if have.get(rel) == sz:
                continue
            s3.upload_file(p, bucket, f"{prefix}/{rel}")
            n += 1
            nb += sz
    elif mode == "down":
        bucket, prefix = split(src)
        objs = listing(s3, bucket, prefix)
        if not objs:  # a single object: dst is then the target FILE path
            sz = s3.head_object(Bucket=bucket, Key=prefix)["ContentLength"]
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            if force or not os.path.exists(dst) or os.path.getsize(dst) != sz:
                s3.download_file(bucket, prefix, dst + ".part")
                os.replace(dst + ".part", dst)
                n, nb = 1, sz
            objs = {}
        for rel, sz in sorted(objs.items()):
            p = os.path.join(dst, rel)
            if not force and os.path.exists(p) and os.path.getsize(p) == sz:
                continue
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            s3.download_file(bucket, f"{prefix}/{rel}", p + ".part")
            os.replace(p + ".part", p)
            n += 1
            nb += sz
    else:
        raise SystemExit(f"mode must be up|down, got {mode}")
    print(f"s3sync {mode} {src} -> {dst}: {n} files, {nb / 1e6:.1f} MB copied", flush=True)


if __name__ == "__main__":
    main()
