"""CPU tests for the MolmoPoint-GUISyn source (mm_olmo ``molmo2_syn_point`` port): reading the
Arrow shards laid out like the copy on weka, the ``gui_point:`` / ``pointing:`` question forms, the box-center
point, element sampling per image and epoch, and the Molmo2-Stage1 v2 pointing group."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from olmo_core.data.multimodal import GuiSynDatasetConfig
from olmo_core.data.multimodal.gui_syn import GUI_POINT_STYLE, GUI_SYN_REVISION
from olmo_core.data.multimodal.pixmo_points_v2 import STAGE1_PROMPT_FAMILY
from olmo_core.data.multimodal.sft_formatter import SftFormatter
from olmo_core.exceptions import OLMoConfigurationError

W, H = 64, 48


class _PromptTok:
    """Minimal tokenizer that records the user turns it templates."""

    eos_token_id = 1
    bos_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.prompts.append(messages[0]["content"])
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


def _el(name, intents, x=32, y=12):
    return {"name": name, "intent": intents, "x_center": x, "y_center": y, "width": 8, "height": 6}


def _write_split(root: Path, part: str, split: str, annotations, tmp_path: Path):
    """One shard of ``split`` laid out like ``GUI_SYN_ROOT``."""
    from datasets import Dataset
    from PIL import Image

    n = len(annotations)
    ds = Dataset.from_dict(
        {
            "id": [f"{part}-{split}-{i}" for i in range(n)],
            "image": [Image.new("RGB", (W, H), color=(10 * i, 30, 30)) for i in range(n)],
            "annotation": annotations,
            "metadata": ["{}"] * n,
            "html": ["<html></html>"] * n,
        }
    )
    tmp = tmp_path / f"_save_{part}_{split}"
    ds.save_to_disk(str(tmp))
    shard_dir = root / part / "0.0.0" / GUI_SYN_REVISION
    shard_dir.mkdir(parents=True, exist_ok=True)
    name = "molmo_point-gui_syn-validation.arrow"
    if split == "train":
        name = "molmo_point-gui_syn-train-00000-of-00001.arrow"
    shutil.copy(next(tmp.glob("*.arrow")), shard_dir / name)


def _cache(tmp_path, train=None, validation=None, part="web"):
    cache = tmp_path / "gui_syn"
    train = train or [
        [_el("Close Google tab button", ["Close the Google tab", "Click the X to close Google"])],
        # No usable element at all: the image is never trained on.
        [_el("", []), _el("  ", [])],
        # One usable element among unusable ones.
        [_el("", []), _el("Search Box", ["Type in the search box"], x=16, y=36)],
    ]
    _write_split(cache, part, "train", train, tmp_path)
    if validation is not None:
        _write_split(cache, part, "validation", validation, tmp_path)
    return str(cache)


def _cfg(cache, **kw):
    kw.setdefault("parts", ("web",))
    kw.setdefault("max_crops", 1)
    return GuiSynDatasetConfig(dataset_path=cache, **kw)


def _turns(ds, i, epoch=0):
    ds.set_epoch(epoch)
    row = ds._data[int(ds._index[i])]
    rng = ds.epoch_rng(i)
    messages = ds.format_row(row["annotation"], row["image"].size, rng)
    fmt = SftFormatter(seed=0, **STAGE1_PROMPT_FAMILY)
    return [fmt.format_turns(m, index=i, rng=rng)[0] for m in messages]


def test_gui_point_tag_follows_the_prompt_family():
    """Stage 1 prefixes ``gui_point:`` like ``cosyn_point:``; the stage-2 family adds no prefix,
    as for every demo pointing style."""
    assert SftFormatter(**STAGE1_PROMPT_FAMILY).style_prefix(GUI_POINT_STYLE) == "gui_point:"
    assert SftFormatter().style_prefix(GUI_POINT_STYLE) == ""


def test_intents_get_their_own_tag_and_names_the_pointing_tag(tmp_path):
    cache = _cache(tmp_path)
    by_intent = _turns(_cfg(cache, p_intent=1.0).build(None), 0)
    assert by_intent[0][0] in (
        "gui_point: Close the Google tab",
        "gui_point: Click the X to close Google",
    )
    # The answer is labelled with the lowercased element name, never the intent.
    assert by_intent[0][1].endswith(">close google tab button</points>")

    by_name = _turns(_cfg(cache, p_intent=0.0).build(None), 0)
    assert by_name == [
        ("pointing: close google tab button", by_name[0][1]),
    ]
    assert by_name[0][1].endswith(">close google tab button</points>")


def test_an_element_without_a_name_is_asked_by_its_intent(tmp_path):
    cache = _cache(tmp_path, train=[[_el("", ["Open the settings menu"])]])
    (user, answer) = _turns(_cfg(cache, p_intent=0.0).build(None), 0)[0]
    assert user == "gui_point: Open the settings menu"
    assert answer.endswith(">open the settings menu</points>")


def test_the_point_is_the_box_center(tmp_path):
    """x_center / width and y_center / height, in the html-v2 thousandths."""
    cache = _cache(tmp_path)
    ds = _cfg(cache, p_intent=0.0).build(None)
    assert _turns(ds, 0)[0][1] == '<points coords="1 1 500 250">close google tab button</points>'
    assert _turns(ds, 1)[0][1] == '<points coords="1 1 250 750">search box</points>'


def test_unusable_elements_and_images_are_skipped(tmp_path):
    cache = _cache(tmp_path)
    ds = _cfg(cache).build(None)
    assert len(ds._data) == 3 and len(ds) == 2
    assert len(_turns(ds, 1)) == 1  # the empty element next to "Search Box" is dropped


def test_reads_only_the_train_split(tmp_path):
    cache = _cache(tmp_path, validation=[[_el("Held out", ["Click the held-out button"])]])
    ds = _cfg(cache).build(None)
    assert len(ds._data) == 3
    assert "held out" not in str([_turns(ds, i) for i in range(len(ds))])


def _crowded_cache(tmp_path, n=40):
    elements = [_el(f"Button {k}", [f"Press button {k}"], x=k, y=k) for k in range(n)]
    return _cache(tmp_path, train=[elements])


def test_samples_at_most_max_elements_per_image(tmp_path):
    ds = _cfg(_crowded_cache(tmp_path), max_elements=16, p_intent=0.0).build(None)
    turns = _turns(ds, 0)
    assert len(turns) == 16
    assert len({user for user, _ in turns}) == 16  # drawn without replacement
    small = _cfg(_crowded_cache(tmp_path / "small", n=5), max_elements=16).build(None)
    assert len(_turns(small, 0)) == 5


def test_element_sample_rotates_across_epochs_and_is_deterministic(tmp_path):
    ds = _cfg(_crowded_cache(tmp_path), max_elements=16, p_intent=0.0).build(_PromptTok())
    samples = [frozenset(u for u, _ in _turns(ds, 0, epoch=e)) for e in range(4)]
    assert len(set(samples)) > 1, "the element sample never rotated across epochs"
    assert samples[2] == frozenset(u for u, _ in _turns(ds, 0, epoch=2))

    ds.set_epoch(1)
    a = ds[0]
    np.testing.assert_array_equal(a["input_ids"], ds[0]["input_ids"])
    other = _cfg(_crowded_cache(tmp_path / "s1"), max_elements=16, p_intent=0.0, seed=1).build(None)
    other.set_epoch(1)
    assert frozenset(u for u, _ in _turns(other, 0, epoch=1)) != samples[1]


def test_example_end_to_end(tmp_path):
    cache = _cache(tmp_path)
    tok = _PromptTok()
    ds = _cfg(cache, p_intent=1.0).build(tok)
    ex = ds[0]
    assert len(ex["input_ids"]) == len(ex["loss_masks"]) and ex["loss_masks"].sum() > 0
    assert any(p.startswith("gui_point: ") for p in tok.prompts)


def test_missing_data_is_a_clear_error(tmp_path):
    with pytest.raises(OLMoConfigurationError, match="No 'train' shards"):
        _cfg(str(tmp_path / "nowhere")).build(None)


@pytest.mark.parametrize(
    "kw",
    [
        dict(parts=()),
        dict(parts=("tablet",)),
        dict(parts=("web", "web")),
        dict(max_elements=0),
        dict(p_intent=1.5),
    ],
)
def test_config_validation(tmp_path, kw):
    with pytest.raises(OLMoConfigurationError):
        GuiSynDatasetConfig(dataset_path=str(tmp_path), **kw).build(None)


# ---------------------------------------------------------------------------
# Molmo2-Stage1.py wiring
# ---------------------------------------------------------------------------


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    path = Path(__file__).parents[3] / "scripts" / "train" / "Molmo2-Stage1.py"
    spec = importlib.util.spec_from_file_location("_stage1_gui", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_gui"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_v2_pointing_group_includes_gui_syn():
    mod = _load_stage1_module()
    assert mod.GUI_POINTING is True
    fields = {f.name for f in mod.ExperimentConfig.__dataclass_fields__.values()}
    assert {"gui_syn", "gui_pointing"} <= fields

    gui = GuiSynDatasetConfig()
    config = SimpleNamespace(pointing_v2="p", count_v2="c", gui_syn=gui, gui_pointing=True)
    names = [name for name, _ in mod._v2_pointing_sources(config)]
    assert names == ["pixmo_points_v2", "pixmo_count_v2", "cosyn_point_v2", "gui_syn"]
    assert dict(mod._v2_pointing_sources(config))["gui_syn"] is gui

    config.gui_pointing = False
    names = [name for name, _ in mod._v2_pointing_sources(config)]
    assert names == ["pixmo_points_v2", "pixmo_count_v2", "cosyn_point_v2"]
