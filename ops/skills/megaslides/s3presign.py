"""Stdlib S3 upload + presigned GET (SigV4), so figures get a URL Google Slides can fetch.

No boto3/aws CLI on the login node. Reads keys from ~/.aws/credentials (profile `S3`, the
one PAIL/OLMo-core tooling uses). Google copies the image at insert time, so a short expiry
is enough; the object itself stays private.
"""
import configparser
import datetime
import hashlib
import hmac
import os
import urllib.parse
import urllib.request

BUCKET = "ai2-llm"
REGION = "us-east-1"
PREFIX = "checkpoints/prasanns/megaslides"
HOST = f"{BUCKET}.s3.amazonaws.com"


def _creds(profile="S3"):
    c = configparser.ConfigParser()
    c.read(os.path.expanduser("~/.aws/credentials"))
    return c[profile]["aws_access_key_id"], c[profile]["aws_secret_access_key"]


def _sign(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _signing_key(secret, date):
    k = _sign(("AWS4" + secret).encode(), date)
    for part in (REGION, "s3", "aws4_request"):
        k = _sign(k, part)
    return k


def _quote_key(key):
    return urllib.parse.quote(key, safe="/-_.~")


def upload(path, content_type="image/png"):
    """PUT the file under PREFIX/<sha12>.<ext>; returns the object key (idempotent by content)."""
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    key = f"{PREFIX}/{sha[:12]}{os.path.splitext(path)[1].lower()}"
    ak, sk = _creds()
    now = datetime.datetime.now(datetime.timezone.utc)
    amzdate, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    headers = {"content-type": content_type, "host": HOST, "x-amz-content-sha256": sha, "x-amz-date": amzdate}
    signed = ";".join(sorted(headers))
    canon = "\n".join(["PUT", "/" + _quote_key(key), "",
                       "".join(f"{h}:{headers[h]}\n" for h in sorted(headers)), signed, sha])
    scope = f"{date}/{REGION}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, hashlib.sha256(canon.encode()).hexdigest()])
    sig = hmac.new(_signing_key(sk, date), sts.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={signed}, Signature={sig}"
    req = urllib.request.Request(f"https://{HOST}/{_quote_key(key)}", data=data, method="PUT",
                                 headers={k: v for k, v in headers.items() if k != "host"})
    urllib.request.urlopen(req, timeout=60).read()
    return key


def presign(key, expires=3600):
    ak, sk = _creds()
    now = datetime.datetime.now(datetime.timezone.utc)
    amzdate, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{date}/{REGION}/s3/aws4_request"
    q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{ak}/{scope}",
         "X-Amz-Date": amzdate, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": "host"}
    qs = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in sorted(q.items()))
    canon = "\n".join(["GET", "/" + _quote_key(key), qs, f"host:{HOST}\n", "host", "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, hashlib.sha256(canon.encode()).hexdigest()])
    sig = hmac.new(_signing_key(sk, date), sts.encode(), hashlib.sha256).hexdigest()
    return f"https://{HOST}/{_quote_key(key)}?{qs}&X-Amz-Signature={sig}"


def public_url(path, expires=3600):
    return presign(upload(path), expires)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(public_url(p))
