"""Lossless fast-tokenizer export; never infer encoding semantics from a model class."""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils.hub import cached_file

PROBES = (
    "9078563412",
    "12345678901234567890",
    " 123 4567\n89012345",
    "def f(x):\n    return x + 123456\n",
    "Hello, world! Don't split incorrectly.",
    "é e\u0301 中文 العربية 🙂",
    "\t\n  spaces\r\n",
    "<|im_start|>assistant\n<think>",
)


def tokenizer_source(checkpoint, identifier, override=None):
    """Resolve explicit override before checkpoint-side tokenizer, then saved identifier."""
    if override is not None:
        return str(override)
    nearby = Path(checkpoint).parent / "tokenizer" if checkpoint is not None else None
    if nearby is not None and nearby.is_dir():
        return str(nearby)
    return identifier


def _semantics(backend):
    value = json.loads(backend.to_str())
    # HF enables/disables these per call; they are not tokenizer model semantics.
    value.pop("padding", None)
    value.pop("truncation", None)
    # Transformers 5 materializes the tokenizers default pair/type-ID behavior.
    # This exact identity processor adds no tokens; do not permit any other rewrite.
    identity = {
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {},
    }
    if value.get("post_processor") == identity:
        value["post_processor"] = None
    return value


def validate_tokenizer_backend(tokenizer, reference):
    """Reject any backend change, even when vocabulary and special-token IDs match."""
    if _semantics(tokenizer.backend_tokenizer) != _semantics(reference):
        raise ValueError("Tokenizer backend changed (model/normalizer/pre-tokenizer/decoder/etc.)")
    for text in PROBES:
        for special in (False, True):
            expected = reference.encode(text, add_special_tokens=special).ids
            actual = tokenizer.encode(text, add_special_tokens=special)
            if actual != expected:
                raise ValueError(f"Tokenizer encode mismatch for {text!r}, special={special}")
            pair = reference.encode(text, pair="1234567", add_special_tokens=special)
            actual_pair = tokenizer(text, text_pair="1234567", add_special_tokens=special)
            if actual_pair["input_ids"] != pair.ids:
                raise ValueError(f"Tokenizer pair mismatch for {text!r}")
            if "token_type_ids" in actual_pair and actual_pair["token_type_ids"] != pair.type_ids:
                raise ValueError(f"Tokenizer pair type IDs mismatch for {text!r}")
            if tokenizer.decode(actual, skip_special_tokens=False) != reference.decode(
                expected, skip_special_tokens=False
            ):
                raise ValueError(f"Tokenizer decode mismatch for {text!r}")


def load_tokenizer_losslessly(source, *, revision=None, cache_dir=None):
    """Load the serialized backend with the generic fast class, preserving chat metadata.

    A tokenizer.json is required. Slow-tokenizer reconstruction and model-specific regex
    rewriting are intentionally not allowed in checkpoint conversion.
    """
    source = str(source)
    path = cached_file(source, "tokenizer.json", revision=revision, cache_dir=cache_dir)
    if path is None:
        raise ValueError(f"No serialized tokenizer.json at {source}")
    reference = Tokenizer.from_file(path)
    # Do not dispatch via AutoTokenizer here: model-specific classes can replace the
    # pre-tokenizer even when loading an existing tokenizer.json.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        source, revision=revision, cache_dir=cache_dir, clean_up_tokenization_spaces=False
    )
    validate_tokenizer_backend(tokenizer, reference)
    return tokenizer, reference, path


def save_tokenizer_losslessly(tokenizer, reference, output, *, source, revision=None):
    """Save and independently AutoTokenizer-reload before recording successful export."""
    output = Path(output)
    validate_tokenizer_backend(tokenizer, reference)
    tokenizer.save_pretrained(output)
    reloaded = AutoTokenizer.from_pretrained(output, local_files_only=True)
    validate_tokenizer_backend(reloaded, reference)
    for attr in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "model_max_length",
        "chat_template",
    ):
        if getattr(reloaded, attr) != getattr(tokenizer, attr):
            raise ValueError(f"Tokenizer metadata changed on reload: {attr}")
    receipt = {
        "passed": True,
        "source": str(source),
        "revision": revision,
        "backend_sha256": hashlib.sha256(
            json.dumps(_semantics(reference), sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest(),
        "tokenizer_class": type(reloaded).__name__,
        "probe_count": len(PROBES),
    }
    (output / "tokenizer-export-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return reloaded


def export_checkpoint_tokenizer(
    checkpoint, output, config, *, override=None, revision=None, max_sequence_length=None
):
    """Export tokenizer without changing its text encoding or silently overriding its source."""
    source = tokenizer_source(checkpoint, config.identifier, override)
    if source is None:
        raise ValueError("No tokenizer source; pass --tokenizer to produce a usable HF checkpoint")
    tokenizer, reference, _ = load_tokenizer_losslessly(source, revision=revision)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary size {len(tokenizer)} != {config.vocab_size}")
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        expected = getattr(config, attr)
        # An unspecified optional ID is not an instruction to discard source metadata.
        # Explicit IDs in the training config remain authoritative.
        if expected is not None:
            if expected not in tokenizer.get_vocab().values():
                raise ValueError(f"{attr}={expected} not in tokenizer vocabulary")
            setattr(tokenizer, attr, expected)
    if max_sequence_length is not None:
        tokenizer.model_max_length = max_sequence_length
    tokenizer = save_tokenizer_losslessly(
        tokenizer, reference, output, source=source, revision=revision
    )
    return tokenizer
