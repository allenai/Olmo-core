import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    MuonConfig,
    MuPConfig,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.optim.mup import mup_lr_multipliers, split_groups_by_lr_multiplier
from olmo_core.utils import log_once

D_MODEL = 128
HIDDEN = 512
VOCAB = 256
BASE_WIDTH = 64


def _model(
    init_method: InitMethod = InitMethod.nanochat,
    d_model: int = D_MODEL,
    hidden: int = HIDDEN,
    n_layers: int = 2,
) -> nn.Module:
    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=VOCAB,
        n_layers=n_layers,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=hidden, bias=False),
        init_method=init_method,
    ).build(init_device="cpu")


def _base_model() -> nn.Module:
    """Half the width and depth of :func:`_model`."""
    return _model(d_model=D_MODEL // 2, hidden=HIDDEN // 2, n_layers=1)


def _group_of(optim: torch.optim.Optimizer, param: torch.Tensor) -> dict:
    (group,) = [g for g in optim.param_groups if any(p is param for p in g["params"])]
    return group


def test_mup_lr_multipliers():
    model = _model()
    multipliers = mup_lr_multipliers(
        model, MuPConfig(base_width=BASE_WIDTH, embedding_lr_multiplier=3.0)
    )
    assert multipliers["embeddings.weight"] == 3.0
    assert multipliers["blocks.0.attention.w_q.weight"] == BASE_WIDTH / D_MODEL
    assert multipliers["blocks.0.feed_forward.w2.weight"] == BASE_WIDTH / HIDDEN
    assert multipliers["lm_head.w_out.weight"] == BASE_WIDTH / D_MODEL
    assert multipliers["blocks.0.attention_norm.weight"] == 1.0


def test_mup_config_rejects_nonpositive_base_width():
    with pytest.raises(OLMoConfigurationError, match="base_width"):
        MuPConfig(base_width=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({}, "exactly one", id="neither"),
        pytest.param({"base_width": 8, "base_fan_ins": {"a": 8}}, "exactly one", id="both"),
        pytest.param({"base_fan_ins": {}}, "must not be empty", id="empty"),
        pytest.param({"base_fan_ins": {"a": 0}}, r"base_fan_ins\['a'\]", id="nonpositive"),
    ],
)
def test_mup_config_rejects_invalid_bases(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        MuPConfig(**kwargs)


def test_mup_base_model_multipliers():
    config = MuPConfig.from_base_model(_base_model(), embedding_lr_multiplier=3.0)
    assert MuPConfig.from_dict(config.as_config_dict()) == config

    # Every parameter of the base model gets the base learning rate.
    for name, multiplier in mup_lr_multipliers(_base_model(), config).items():
        assert multiplier == (3.0 if name == "embeddings.weight" else 1.0), name

    # A wider, deeper model scales each matrix by how much its own fan-in grew.
    multipliers = mup_lr_multipliers(_model(), config)
    assert multipliers["embeddings.weight"] == 3.0
    for idx in ("0", "1"):
        assert multipliers[f"blocks.{idx}.attention.w_q.weight"] == 0.5
        assert multipliers[f"blocks.{idx}.feed_forward.w2.weight"] == 0.5
        assert multipliers[f"blocks.{idx}.attention_norm.weight"] == 1.0
    assert multipliers["lm_head.w_out.weight"] == 0.5


def test_mup_base_model_param_groups():
    base = _base_model()
    optim = AdamWConfig(lr=1e-3, mup=MuPConfig.from_base_model(base)).build(base)

    (group,) = optim.param_groups
    assert len(group["params"]) == len(list(base.parameters()))
    assert group["lr"] == pytest.approx(1e-3)
    assert group["lr_multiplier"] == 1.0


def test_mup_base_model_rejects_unknown_roles():
    config = MuPConfig(base_fan_ins={"lm_head.w_out.weight": D_MODEL})
    with pytest.raises(OLMoConfigurationError, match="no matrix with role 'blocks"):
        mup_lr_multipliers(_model(), config)


@pytest.mark.parametrize("config_cls", [AdamWConfig, SkipStepAdamWConfig])
def test_mup_param_groups(config_cls):
    model = _model()
    config = config_cls(lr=1e-3, mup=MuPConfig(base_width=BASE_WIDTH))
    # The learning rate is read when the optimizer is built, not when the config is created.
    config.lr = 2e-3
    optim = config.build(model)

    multipliers = mup_lr_multipliers(model, MuPConfig(base_width=BASE_WIDTH))
    assert sum(len(g["params"]) for g in optim.param_groups) == len(list(model.parameters()))
    for name, param in model.named_parameters():
        group = _group_of(optim, param)
        assert group["lr_multiplier"] == multipliers[name]
        assert group["lr"] == pytest.approx(2e-3 * multipliers[name])
        assert group["initial_lr"] == group["lr"]
    assert len(optim.param_groups) == len(set(multipliers.values()))


def test_mup_composes_with_group_overrides():
    model = _model()
    optim = AdamWConfig(
        lr=1e-3,
        weight_decay=0.1,
        mup=MuPConfig(base_width=BASE_WIDTH),
        group_overrides=[
            OptimGroupOverride(params=["*.w2.weight"], opts={"lr": 4e-3, "betas": (0.8, 0.9)}),
            OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
        ],
    ).build(model)

    w2 = _group_of(optim, model.blocks["0"].feed_forward.w2.weight)
    assert w2["lr"] == pytest.approx(4e-3 * BASE_WIDTH / HIDDEN)
    assert w2["betas"] == (0.8, 0.9)
    embeddings = _group_of(optim, model.embeddings.weight)
    assert embeddings["lr"] == pytest.approx(1e-3)
    assert embeddings["weight_decay"] == 0.0
    w1 = _group_of(optim, model.blocks["0"].feed_forward.w1.weight)
    assert w1["lr"] == pytest.approx(1e-3 * BASE_WIDTH / D_MODEL)
    assert w1["weight_decay"] == 0.1


@pytest.mark.parametrize("scale_weight_decay", [False, True])
def test_mup_weight_decay(scale_weight_decay: bool):
    model = _model()
    optim = AdamWConfig(
        lr=1e-3,
        weight_decay=0.1,
        mup=MuPConfig(base_width=BASE_WIDTH, scale_weight_decay=scale_weight_decay),
        group_overrides=[OptimGroupOverride(params=["*norm*"], opts={"weight_decay": 0.0})],
    ).build(model)

    w2 = _group_of(optim, model.blocks["0"].feed_forward.w2.weight)
    multiplier = BASE_WIDTH / HIDDEN
    expected = 0.1 / multiplier if scale_weight_decay else 0.1
    assert w2["weight_decay"] == pytest.approx(expected)
    if scale_weight_decay:
        assert w2["lr"] * w2["weight_decay"] == pytest.approx(1e-3 * 0.1)
    assert _group_of(optim, model.lm_head.norm.weight)["weight_decay"] == 0.0


def test_mup_scheduler_keeps_ratios():
    model = _model()
    optim = AdamWConfig(lr=1e-3, mup=MuPConfig(base_width=BASE_WIDTH)).build(model)
    scheduler = CosWithWarmup(warmup=10, alpha_f=0.1)
    for step in (0, 5, 10, 50, 100):
        trainer = SimpleNamespace(global_step=step, max_steps=100)
        for group in optim.param_groups:
            scheduler.set_lr(group, trainer)  # type: ignore[arg-type]
        expected = scheduler.get_lr(1e-3, step, 100)
        for group in optim.param_groups:
            assert group["lr"] == pytest.approx(expected * group["lr_multiplier"])


def test_mup_split_requires_every_param():
    group = OptimGroupOverride(params=["a", "b"], opts={})
    with pytest.raises(OLMoConfigurationError, match="'b'"):
        split_groups_by_lr_multiplier([group], {"a": 1.0}, MuPConfig(base_width=8), lr=1e-3)


def test_mup_rejected_for_matrix_aware_optimizers():
    with pytest.raises(OLMoConfigurationError, match="not supported for MuonConfig"):
        MuonConfig(mup=MuPConfig(base_width=BASE_WIDTH)).build(_model())


def test_mup_warns_without_nanochat_init(caplog):
    log_once.cache_clear()
    with caplog.at_level(logging.WARNING):
        mup_lr_multipliers(_model(InitMethod.normal), MuPConfig(base_width=BASE_WIDTH))
    assert "uses init method 'normal'" in caplog.text
