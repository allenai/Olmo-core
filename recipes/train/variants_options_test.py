"""Every softtoken (and layer-drop) flag the trainer parses must reach the variant.

``Variant.options_from_namespace`` copies only the option dataclass's FIELDS from the argparse namespace;
a ``--st-*`` flag without a matching field is parsed and then silently dropped. On 2026-10-22 four flags
(``--st-kept-doc-k``, ``--st-keep-token-min``, ``--st-pad-multiple``, ``--st-span-len``) were dropped this
way, and two training rounds ran without the rule they were launched to test. Static, no torch import.

    python -m pytest -q recipes/train/variants_options_test.py
"""

import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = os.path.join(HERE, "variants.py")
TRAINER = os.path.join(HERE, "research_trainer", "train_ctc_suite.py")


def _option_fields(cls_name):
    tree = ast.parse(open(VARIANTS).read())
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    def fields(name):
        node = classes[name]
        out = {s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                out |= fields(base.id)
        return out

    return fields(cls_name)


def _softtoken_fields():
    return _option_fields("SoftTokenOptions")


def test_every_trainer_st_flag_is_a_softtoken_option_field():
    flags = re.findall(r'add_argument\(\s*"--(st-[a-z0-9-]+)"', open(TRAINER).read())
    assert flags, "no --st-* flags found; did the trainer move?"
    missing = sorted(f for f in flags if f.replace("-", "_") not in _softtoken_fields())
    assert not missing, f"--st-* flags parsed by the trainer but dropped by options_from_namespace: {missing}"


def test_softtoken_variant_reads_only_declared_fields():
    src = open(VARIANTS).read()
    used = set(re.findall(r"\bo\.(st_[a-z0-9_]+)", src)) | set(re.findall(r'getattr\(o, "(st_[a-z0-9_]+)"', src))
    missing = sorted(used - _softtoken_fields())
    assert not missing, f"variants.py reads softtoken options that are not dataclass fields: {missing}"


def test_every_trainer_layer_drop_flag_is_a_layerdrop_option_field():
    flags = re.findall(r'add_argument\(\s*"--((?:layer-drop|train-skip)-[a-z0-9-]+)"', open(TRAINER).read())
    assert "train-skip-layers" in flags and "layer-drop-p" in flags, flags
    fields = _option_fields("LayerDropOptions")
    missing = sorted(f for f in flags if f.replace("-", "_") not in fields)
    assert not missing, f"layer-drop flags parsed by the trainer but dropped by options_from_namespace: {missing}"
    used = set(re.findall(r"\bo\.((?:layer_drop|train_skip)_[a-z0-9_]+)", open(VARIANTS).read()))
    assert not (used - fields), f"variants.py reads layer-drop options that are not fields: {sorted(used - fields)}"


def test_no_variant_defines_a_method_twice():
    """A second ``def`` of the same name in a class silently replaces the first. On 2026-10-24 SoftToken had two
    ``export_config`` methods; the later one won, so the ``early_lora`` block never reached config.json and every eval
    loaded the LoRA checkpoints as plain models (a DCP load reads only the keys the model asks for: the adapters were
    dropped without an error)."""
    tree = ast.parse(open(VARIANTS).read())
    dups = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        seen = set()
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in seen:
                    dups.append(f"{cls.name}.{node.name}")
                seen.add(node.name)
    assert not dups, dups
