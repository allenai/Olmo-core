"""Model-only copy of an olmo-core ``model_and_optim`` checkpoint as ONE safetensors file (olmo state-dict keys), for
moving weka checkpoints to the Berkeley cluster via S3. Uses ``unshard_checkpoint(optim=False)``, which reads the
distcp directly and never builds the model -- so it needs neither flash-linear-attention nor a GPU (building the
Qwen3.5 hybrid asserts ``has_fla()``, which the stock Beaker image lacks). Non-distributed, CPU.

    python resave_model_only.py /weka/.../stepN/model_and_optim /tmp/out/<name>

Writes ``<out>/model.safetensors`` + ``<out>/.resave_done``; load with ``grad_importance.py --ckpt-format olmo_st``.
"""
import os
import sys
import time


def main():
    src, out = sys.argv[1], sys.argv[2]
    from olmo_core.distributed.checkpoint import unshard_checkpoint

    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    model_path, _ = unshard_checkpoint(src, out, optim=False, use_safetensors=True, save_overwrite=True)
    print(f"unsharded model of {src} -> {model_path} ({os.path.getsize(model_path) / 2**30:.1f} GiB, {time.time() - t0:.0f}s)",
          flush=True)
    if os.path.basename(str(model_path)) != "model.safetensors":
        os.replace(model_path, os.path.join(out, "model.safetensors"))
    open(os.path.join(out, ".resave_done"), "w").write(src + "\n")


if __name__ == "__main__":
    main()
