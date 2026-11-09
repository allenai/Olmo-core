"""Download resave_model_only.py outputs from S3 onto node-local /data (Berkeley compute nodes have egress + boto3).

    $PY fetch_s3_ckpt.py s3://ai2-llm/checkpoints/prasanns/_transfer/gradprobe/ckpts setA32k,setAbase /data/prasann/devloss_grid/ckpts

Each <prefix>/<name>/{model.safetensors,.resave_done} -> <dst>/<name>/. Credentials: ~/.aws (profile S3). Skips files
whose size already matches.
"""
import os
import sys
import time

import boto3


def main():
    prefix, names, dst = sys.argv[1].rstrip("/"), sys.argv[2].split(","), sys.argv[3]
    bucket, key0 = prefix[len("s3://"):].split("/", 1)
    s3 = boto3.Session(profile_name="S3").client("s3")
    for n in names:
        os.makedirs(os.path.join(dst, n), exist_ok=True)
        for f in ("model.safetensors", ".resave_done"):
            key, out = f"{key0}/{n}/{f}", os.path.join(dst, n, f)
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if os.path.exists(out) and os.path.getsize(out) == size:
                print(f"skip {out} (complete)", flush=True)
                continue
            t0 = time.time()
            s3.download_file(bucket, key, out + ".part")
            os.replace(out + ".part", out)
            print(f"{key} -> {out}: {size / 2**30:.1f} GiB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
