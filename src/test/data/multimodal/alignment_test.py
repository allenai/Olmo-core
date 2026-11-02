from dataclasses import dataclass, field

import pytest

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import (
    MultimodalMixtureConfig,
    MultimodalSourceConfig,
)
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds


@dataclass
class ExampleDatasetConfig(Config):
    size: int = 10
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)

    def build(self, tokenizer):
        return ExampleDataset(self)


class ExampleDataset:
    def __init__(self, config):
        self.config = config
        self.content_fingerprint = "example-dataset"

    def __len__(self):
        return self.config.size

    def __getitem__(self, index):
        return self.get(index, 0)

    def get(self, index, epoch=0):
        return {"row": index, "epoch": epoch}

    def raw_image_references(self, index):
        return (f"image-{index}",)


def test_config_round_trip_and_nested_overrides():
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources={
            "new_captions": MultimodalSourceConfig(
                dataset=PixMoCapDatasetConfig(dataset_path="synthetic", mode="caption")
            )
        },
        target_loss_mass={"new_captions": 1.0},
        mean_loss_weight={"new_captions": 12.0},
    )
    restored = MultimodalMixtureConfig.from_dict(config.as_config_dict())
    assert restored == config
    updated = restored.merge(["sources.new_captions.dataset.max_crops=2"])
    assert isinstance(updated.sources["new_captions"], MultimodalSourceConfig)
    assert isinstance(updated.sources["new_captions"].dataset, PixMoCapDatasetConfig)
    assert updated.sources["new_captions"].dataset.max_crops == 2
    assert config.sources["new_captions"].dataset.max_crops == 8


def test_build_retains_order_and_calibrates_loss_mass():
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources={"z_caption": ExampleDatasetConfig(), "a_count": ExampleDatasetConfig()},
        target_loss_mass={"a_count": 0.25, "z_caption": 0.75},
        mean_loss_weight={"z_caption": 3.0, "a_count": 1.0},
    )
    token_ids = Molmo2TokenIds(im_start_id=7)
    mixture = config.build(tokenizer=object(), token_ids=token_ids)
    assert mixture.names == ["z_caption", "a_count"]
    assert mixture.weights == pytest.approx([0.5, 0.5])
    for dataset in mixture.datasets:
        assert dataset.config.token_ids == token_ids
    assert config.sources["z_caption"].token_ids != token_ids


class _StubTokenizer:
    eos_token_id = 2
    pad_token_id = 3

    def __init__(self):
        self.vocab = {"<|im_end|>": 4}

    def add_tokens(self, tokens, special_tokens=False):
        for token in tokens:
            self.vocab.setdefault(token, len(self.vocab) + 4)

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


@pytest.mark.parametrize("tokenizer_json_split", [True, False])
def test_build_tokenizer_uses_configured_identity(monkeypatch, tokenizer_json_split):
    from transformers import AutoTokenizer
    from transformers.models.auto import tokenization_auto

    import olmo_core.nn.hf.tokenizer as hf_tokenizer

    calls = []

    def load_auto(identifier, **kwargs):
        calls.append(("auto", identifier, kwargs))
        return _StubTokenizer()

    def load_lossless(identifier, **kwargs):
        calls.append(("lossless", identifier, kwargs))
        return _StubTokenizer(), None, None

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_auto)
    monkeypatch.setattr(tokenization_auto, "get_tokenizer_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(hf_tokenizer, "load_tokenizer_losslessly", load_lossless)
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig(
            vocab_size=5, eos_token_id=2, pad_token_id=3, identifier="parent"
        ),
        model_vocab_size=16,
        tokenizer_revision="pinned-revision",
        tokenizer_json_split=tokenizer_json_split,
    )
    tokenizer, ids = config.build_tokenizer()
    assert tokenizer.eos_token_id == 2
    assert ids.im_start_id == 5
    if tokenizer_json_split:
        expected = ("lossless", "parent", {"revision": "pinned-revision", "cache_dir": None})
    else:
        expected = (
            "auto",
            "parent",
            {"config": None, "revision": "pinned-revision", "cache_dir": None, "use_fast": False},
        )
    assert calls[0] == expected
    config.tokenizer.pad_token_id = 8
    with pytest.raises(ValueError, match="pad_token_id"):
        config.build_tokenizer()


def test_tokenizer_json_split_is_the_default():
    config = MultimodalMixtureConfig(tokenizer=TokenizerConfig.dolma2())
    assert config.tokenizer_json_split is True
    assert config.as_config_dict()["tokenizer_json_split"] is True


def test_build_tokenizer_encodes_as_tokenizer_json(tmp_path):
    """A repository that declares ``GPT2Tokenizer`` (as Dolma2 does) but splits digits in groups of
    three in its ``tokenizer.json``: the mixture must encode exactly as ``tokenizer.json`` does."""
    import json

    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    texts = ["The total was 2024.\n\nNext", "12345678 cats", "long_caption: A dog."] * 4
    raw.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=300, special_tokens=["<unk>", "<eos>", "<pad>", "<|im_end|>"]
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False,
    )
    tokenizer.save_pretrained(tmp_path)
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata["tokenizer_class"] = "GPT2Tokenizer"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    reference = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))

    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig(
            vocab_size=len(tokenizer),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            identifier=str(tmp_path),
        ),
        model_vocab_size=len(tokenizer) + 16,
    )
    built, _ = config.build_tokenizer()
    for text in texts[:3]:
        assert built.encode(text, add_special_tokens=False) == reference.encode(text).ids


def test_calibration_is_bounded_reproducible_and_does_not_mutate(monkeypatch):
    class CalibrationDataset:
        def __init__(self, size):
            self.size = size
            self.calls = []

        def __len__(self):
            return self.size

        def get(self, index, epoch):
            self.calls.append((index, epoch))
            return {"loss_masks": [0.0, float(index + 1)]}

    datasets = {"large": CalibrationDataset(100000000), "small": CalibrationDataset(3)}
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(), mean_loss_weight={"old_source": 2.0}
    )
    monkeypatch.setattr(config, "build_tokenizer", lambda: (object(), Molmo2TokenIds()))
    monkeypatch.setattr(config, "build_sources", lambda *args: datasets)
    first = config.estimate_mean_loss_weights(8, seed=41)
    assert first["small"] == 2.0
    assert len(datasets["large"].calls) == len(set(datasets["large"].calls)) == 8
    assert len(datasets["small"].calls) == 3
    assert all(epoch == 0 for _, epoch in datasets["large"].calls)
    first_indices = datasets["large"].calls[:]
    datasets = {name: datasets[name] for name in reversed(datasets)}
    second = config.estimate_mean_loss_weights(8, seed=41)
    assert first == second
    assert datasets["large"].calls[8:] == first_indices
    assert config.mean_loss_weight == {"old_source": 2.0}
