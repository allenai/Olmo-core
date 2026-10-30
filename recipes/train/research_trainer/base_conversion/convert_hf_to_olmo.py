"""Convert any supported HF Qwen3-dense base model -> olmo-core checkpoint.

Generalizes convert_qwen3_4b_to_olmo.py. olmo-core's example CLI only registers
olmo2/olmo3/llama in its arch dict, so we call its convert function directly with the
qwen3 builder resolved from the HF config (olmo_models.py, alongside this file).

Run as a SINGLE-PROCESS step (not under torchrun); needs one GPU for validate=True.
Output is the olmo-core distributed-checkpoint format (config.json + model_and_optim/).

The eos/pad/bos written into the checkpoint's TokenizerConfig sidecar are derived from
the *actual* HF tokenizer — not hardcoded. (The old script hardcoded eos=151645, the
chat <|im_end|> token; the real base-model EOS for Qwen3 is <|endoftext|>=151643, which
is what eos-segmentation needs at train time. The converted weights are unaffected
either way — Qwen3 ties embeddings and the converter handles that — but a correct
sidecar avoids a footgun.)
"""

import argparse
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from olmo_models import build_transformer_config, resolve_olmo_model  # noqa: E402  (same dir)

# convert_checkpoint_from_hf lives in olmo-core's examples/ (not part of the installed package),
# so it needs a sys.path entry. Default to the repo this file lives in; CORPUS_OLMO_SRC overrides
# for other checkouts (e.g. VESSL's /mnt/corpus clone).
import os as _os
from pathlib import Path as _Path

_OLMO_SRC = _os.environ.get("CORPUS_OLMO_SRC", str(_Path(__file__).resolve().parents[4]))
sys.path.insert(0, f"{_OLMO_SRC}/src/examples/huggingface")
from convert_checkpoint_from_hf import convert_checkpoint_from_hf  # noqa: E402

from olmo_core.data.tokenizer import TokenizerConfig  # noqa: E402


def _convert_qwen3_5(local: str, out: str, model_cfg, tok_cfg) -> None:
    """Hybrid Qwen3.5 conversion: transformers cannot load `qwen3_5`, so read config.json +
    safetensors directly and use olmo-core's dedicated convert_qwen3_5_state_from_hf (handles
    the fused-projection splits + drops the vision tower). Saves the olmo-core distcp format.
    Proven in scripts/debug/verify_qwen35_olmo.py (strict load, 0 missing/unexpected). No
    HF-parity validation step (would require a transformers build that understands qwen3_5)."""
    import glob
    import json
    import os
    import types

    from safetensors.torch import load_file
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.io import join_path
    from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf

    raw = json.load(open(os.path.join(local, "config.json")))
    cfg_obj = types.SimpleNamespace(**raw)
    cfg_obj.text_config = types.SimpleNamespace(**raw["text_config"])

    hf_state = {}
    for shard in sorted(glob.glob(os.path.join(local, "*.safetensors"))):
        hf_state.update(load_file(shard))
    print(f"[convert] qwen3_5: loaded {len(hf_state)} HF tensors", flush=True)

    # CRITICAL: the raw Qwen3.5 safetensors bundle the Multi-Token-Prediction head under
    # `mtp.layers.0.*`. convert_qwen3_5_state_from_hf matches `mtp.layers.0.{mlp,*norm}` as decoder
    # block-0's FFN+norms and silently OVERWRITES them (the MTP head has ONLY layer 0, so EXACTLY
    # block 0 is corrupted, blocks 1-23 fine). This left block-0's MLP at ~random init in the
    # converted base ckpt -> every olmo-Qwen3.5 run trained from a damaged model (learned format,
    # plateaued on content; init masked-CE 6.8 instead of 0.82). transformers' from_pretrained never
    # exposes `mtp.*`, which is why in-memory parity tests missed it. Drop the MTP head before
    # converting. (The vision tower `model.visual.*` is already handled/dropped by the convert fn.)
    _mtp = [k for k in hf_state if k.startswith("mtp.")]
    for k in _mtp:
        del hf_state[k]
    print(f"[convert] qwen3_5: dropped {len(_mtp)} mtp.* (multi-token-prediction) tensors", flush=True)

    model = model_cfg.build(init_device="cpu")
    converted = convert_qwen3_5_state_from_hf(cfg_obj, hf_state)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        sys.exit(f"[convert] qwen3_5 strict-load mismatch: missing={list(missing)[:5]} "
                 f"unexpected={list(unexpected)[:5]}")
    print(f"[convert] qwen3_5: strict load OK ({len(converted)} tensors)", flush=True)

    save_model_and_optim_state(str(join_path(out, "model_and_optim")), model, save_overwrite=True)
    experiment = {"model": model_cfg.as_config_dict(),
                  "dataset": {"tokenizer": tok_cfg.as_config_dict()}}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(experiment, f)
    print(f"[convert] qwen3_5 DONE -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--out", required=True, help="output olmo-core checkpoint dir")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    spec = resolve_olmo_model(args.base_model)
    print(f"[convert] {args.base_model} -> builder={spec.builder} vocab={spec.vocab_size}",
          flush=True)

    # Accept a local model directory (offline-safe) as well as a hub repo id. The hub-cache
    # offline resolver in transformers can fail to find a fully-cached repo by id on air-gapped
    # compute nodes; pointing --base-model at the on-disk snapshot dir sidesteps it entirely.
    local = (
        args.base_model if _os.path.isdir(args.base_model) else snapshot_download(args.base_model)
    )
    print(f"[convert] HF snapshot: {local}", flush=True)

    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    bos = tok.bos_token_id  # may be None for Qwen3 base

    model_cfg = build_transformer_config(spec)
    tok_cfg = TokenizerConfig(
        vocab_size=spec.vocab_size, eos_token_id=eos, pad_token_id=pad,
        bos_token_id=bos, identifier=args.base_model,
    )

    # Hybrid Qwen3.5: transformers can't load `qwen3_5`, so convert directly from files
    # (no HF-parity validation possible). Dense Qwen3 uses the standard olmo-core path.
    if spec.is_hybrid:
        print(f"[convert] eos={eos} pad={pad} bos={bos}; hybrid qwen3_5 (no validate)", flush=True)
        _convert_qwen3_5(local, args.out, model_cfg, tok_cfg)
        return

    print(f"[convert] eos={eos} pad={pad} bos={bos}; validate={not args.no_validate}",
          flush=True)
    convert_checkpoint_from_hf(
        hf_checkpoint_path=local,
        output_path=args.out,
        transformer_config_dict=model_cfg.as_config_dict(),
        tokenizer_config_dict=tok_cfg.as_config_dict(),
        validate=not args.no_validate,
        # CPU fallback (a GPU-less gantry job): conversion and the HF-parity check both run on CPU, just slower
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        validation_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(f"[convert] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
