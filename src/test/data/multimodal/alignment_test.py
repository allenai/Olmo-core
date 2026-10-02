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


def test_build_tokenizer_uses_configured_identity(monkeypatch):
    from transformers import AutoTokenizer
    from transformers.models.auto import tokenization_auto

    class Tokenizer:
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

    calls = []

    def load(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return Tokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(tokenization_auto, "get_tokenizer_config", lambda *args, **kwargs: {})
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig(
            vocab_size=5, eos_token_id=2, pad_token_id=3, identifier="parent"
        ),
        model_vocab_size=16,
        tokenizer_revision="pinned-revision",
    )
    tokenizer, ids = config.build_tokenizer()
    assert tokenizer.eos_token_id == 2
    assert ids.im_start_id == 5
    assert calls[0] == (
        "parent",
        {"config": None, "revision": "pinned-revision", "cache_dir": None, "use_fast": False},
    )
    config.tokenizer.pad_token_id = 8
    with pytest.raises(ValueError, match="pad_token_id"):
        config.build_tokenizer()


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
