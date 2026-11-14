"""Strip a multimodal Olmo 3.5 HF export (``Olmo3MoeVisionForConditionalGeneration``) to its
language model (``Olmo3MoeForCausalLM``), for the text eval battery.

Keeps every ``language_model.*`` tensor bit for bit (prefix removed), writes ``text_config`` as
the config, and copies the text model's remote code and the tokenizer files.

    python olmo35_mm_strip_vision.py MM_HF_DIR TEXT_HF_DIR
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "language_model."
COPY = (
    "configuration_olmo3moe.py",
    "modeling_olmo3moe.py",
    "moe_numerics.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tensors, skipped = {}, 0
    with safe_open(src / "model.safetensors", framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            if key.startswith(PREFIX):
                tensors[key[len(PREFIX) :]] = f.get_tensor(key)
            else:
                skipped += 1
    save_file(tensors, tmp / "model.safetensors", metadata=metadata or {"format": "pt"})
    config = json.loads((src / "config.json").read_text())
    text = dict(config["text_config"])
    for key in ("dtype", "torch_dtype", "transformers_version", "bos_token_id", "eos_token_id", "pad_token_id"):
        if key in config and key not in text:
            text[key] = config[key]
    (tmp / "config.json").write_text(json.dumps(text, indent=2) + "\n")
    for name in COPY:
        if (src / name).exists():
            shutil.copy2(src / name, tmp / name)
    digest = hashlib.sha256((tmp / "model.safetensors").read_bytes()).hexdigest()
    (tmp / "STRIP_RECEIPT.json").write_text(
        json.dumps(
            {"source": str(src), "lm_tensors": len(tensors), "vision_connector_tensors": skipped, "model_sha256": digest},
            indent=2,
        )
        + "\n"
    )
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print("STRIPPED", len(tensors), "kept,", skipped, "dropped ->", out, digest[:16])


if __name__ == "__main__":
    main()
