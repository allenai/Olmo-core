"""
Tests for the optimizer side of CompletedP: per-group lr, eps, weight decay and betas.
"""

from types import SimpleNamespace
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.transformer import CompletedPConfig, TransformerConfig
from olmo_core.optim import (
    AdamWConfig,
    CompletedPOptimConfig,
    CosWithWarmup,
    LionConfig,
    MuonConfig,
    MuPConfig,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.optim.completedp import completedp_betas, completedp_multipliers
from olmo_core.optim.mup import mup_lr_multipliers

D_MODEL = 256
N_LAYERS = 8
BASE_WIDTH = 128
BASE_DEPTH = 2
LR, EPS, WEIGHT_DECAY, BETAS = 1e-3, 1e-8, 0.1, (0.9, 0.95)


def _model(
    d_model: int = D_MODEL, n_layers: int = N_LAYERS, alpha: float = 1.0, completedp: bool = True
) -> nn.Module:
    norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=512,
        n_layers=n_layers,
        n_heads=d_model // 64,
        feed_forward=FeedForwardConfig(hidden_size=4 * d_model, bias=False),
        qk_norm=True,
        use_head_qk_norm=True,
        attn_backend=AttentionBackendName.torch,
        embedding_norm=norm,
        completedp=(
            CompletedPConfig(base_width=BASE_WIDTH, base_depth=BASE_DEPTH, alpha=alpha)
            if completedp
            else None
        ),
    ).build(init_device="cpu")


def _group_of(optim: torch.optim.Optimizer, param: torch.Tensor) -> dict:
    (group,) = [g for g in optim.param_groups if any(p is param for p in g["params"])]
    return group


def _expected_multipliers(
    alpha: float, s: float, d_model: int = D_MODEL, n_layers: int = N_LAYERS
) -> Dict[str, Tuple[float, float, float]]:
    """Table 1 of the paper, written out per parameter of the test model: (lr, eps, wd)."""
    m_n = d_model / BASE_WIDTH
    m_l = n_layers / BASE_DEPTH
    expected = {
        "embeddings.weight": (s, 1 / (m_n * s), s),
        "embedding_norm.weight": (s, 1 / (m_n * s), s),
        "lm_head.norm.weight": (s, 1 / s, s),
        "lm_head.w_out.weight": (s / m_n, 1 / s, m_n * s),
    }
    for i in range(n_layers):
        for name, fan_in in (
            ("attention.w_q", d_model),
            ("attention.w_k", d_model),
            ("attention.w_v", d_model),
            ("attention.w_out", d_model),
            ("feed_forward.w1", d_model),
            ("feed_forward.w3", d_model),
            ("feed_forward.w2", 4 * d_model),
        ):
            r = fan_in / BASE_WIDTH
            expected[f"blocks.{i}.{name}.weight"] = (
                m_l ** (alpha - 1) * s / r,
                m_l**-alpha / (m_n * s),
                r * s,
            )
        for name in ("attention_norm", "feed_forward_norm"):
            expected[f"blocks.{i}.{name}.weight"] = (
                m_l ** (alpha - 1) * s,
                m_l**-alpha / (m_n * s),
                s,
            )
        # Head-wise QK norms share their gain across heads.
        for name in ("attention.q_norm", "attention.k_norm"):
            expected[f"blocks.{i}.{name}.weight"] = (m_l ** (alpha - 1) * s, m_l**-alpha / s, s)
    return expected


@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize(
    "optim_config, s",
    [
        pytest.param(CompletedPOptimConfig(), 1.0, id="base-batch-and-duration"),
        pytest.param(CompletedPOptimConfig(base_batch_size=1, batch_size=4), 2.0, id="4x-batch"),
        pytest.param(
            CompletedPOptimConfig(base_train_tokens=10, train_tokens=40), 0.5, id="4x-tokens"
        ),
    ],
)
def test_completedp_multipliers(alpha: float, optim_config: CompletedPOptimConfig, s: float):
    model = _model(alpha=alpha)
    multipliers = completedp_multipliers(model, optim_config)
    expected = _expected_multipliers(alpha, s)
    assert set(multipliers) == set(expected) == {n for n, _ in model.named_parameters()}
    for name, (lr, eps, weight_decay) in expected.items():
        actual = multipliers[name]
        assert actual["lr"] == pytest.approx(lr), name
        assert actual["eps"] == pytest.approx(eps), name
        assert actual["weight_decay"] == pytest.approx(weight_decay), name


def test_completedp_learning_rates_match_mup_at_base_depth():
    """At the base depth, batch size and duration, CompletedP learning rates are muP's."""
    model = _model(n_layers=BASE_DEPTH)
    completedp = completedp_multipliers(model, CompletedPOptimConfig())
    mup = mup_lr_multipliers(model, MuPConfig(base_width=BASE_WIDTH))
    for name, multiplier in mup.items():
        assert completedp[name]["lr"] == pytest.approx(multiplier), name


@pytest.mark.parametrize("config_cls", [AdamWConfig, SkipStepAdamWConfig])
def test_completedp_param_groups(config_cls):
    model = _model(alpha=0.5)
    optim_config = CompletedPOptimConfig(base_batch_size=1, batch_size=2)
    config = config_cls(
        lr=LR, eps=EPS, betas=BETAS, weight_decay=WEIGHT_DECAY, completedp=optim_config
    )
    # The base values are read when the optimizer is built, not when the config is created.
    config.lr = 2 * LR
    optim = config.build(model)

    multipliers = completedp_multipliers(model, optim_config)
    expected_betas = (1 - 0.1 * 2, 1 - 0.05 * 2)
    assert sum(len(g["params"]) for g in optim.param_groups) == len(list(model.parameters()))
    for name, param in model.named_parameters():
        group = _group_of(optim, param)
        m = multipliers[name]
        assert group["lr"] == pytest.approx(2 * LR * m["lr"])
        assert group["initial_lr"] == group["lr"]
        assert group["eps"] == pytest.approx(EPS * m["eps"])
        assert group["weight_decay"] == pytest.approx(WEIGHT_DECAY * m["weight_decay"])
        assert group["betas"] == pytest.approx(expected_betas)
        for option in ("lr", "eps", "weight_decay"):
            assert group[f"{option}_multiplier"] == m[option]
    distinct = {tuple(m.items()) for m in multipliers.values()}
    assert len(optim.param_groups) == len(distinct)


def test_completedp_composes_with_group_overrides():
    model = _model()
    optim = AdamWConfig(
        lr=LR,
        eps=EPS,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        completedp=CompletedPOptimConfig(base_batch_size=1, batch_size=4),
        group_overrides=[
            OptimGroupOverride(params=["*.w2.weight"], opts={"lr": 4e-3, "betas": (0.8, 0.9)}),
            OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
        ],
    ).build(model)
    multipliers = completedp_multipliers(
        model, CompletedPOptimConfig(base_batch_size=1, batch_size=4)
    )

    w2 = _group_of(optim, model.blocks["0"].feed_forward.w2.weight)
    assert w2["lr"] == pytest.approx(4e-3 * multipliers["blocks.0.feed_forward.w2.weight"]["lr"])
    assert w2["betas"] == pytest.approx((1 - 0.2 * 4, 1 - 0.1 * 4))
    embeddings = _group_of(optim, model.embeddings.weight)
    assert embeddings["weight_decay"] == 0.0
    assert embeddings["lr"] == pytest.approx(LR * 2)
    assert embeddings["betas"] == pytest.approx((1 - 0.1 * 4, 1 - 0.05 * 4))


@pytest.mark.parametrize(
    "optim_config, expected",
    [
        pytest.param(CompletedPOptimConfig(), BETAS, id="base"),
        pytest.param(
            CompletedPOptimConfig(base_batch_size=1, batch_size=4), (0.6, 0.8), id="4x-batch"
        ),
        pytest.param(
            CompletedPOptimConfig(base_train_tokens=1, train_tokens=2),
            (0.95, 0.975),
            id="2x-tokens",
        ),
        pytest.param(
            CompletedPOptimConfig(
                base_batch_size=1, batch_size=2, base_train_tokens=1, train_tokens=2
            ),
            BETAS,
            id="same-steps",
        ),
    ],
)
def test_completedp_betas(optim_config: CompletedPOptimConfig, expected):
    assert completedp_betas(BETAS, optim_config) == pytest.approx(expected)


def test_completedp_betas_unchanged_at_base_ratio():
    assert completedp_betas(BETAS, CompletedPOptimConfig()) == BETAS


def test_completedp_betas_out_of_range():
    with pytest.raises(OLMoConfigurationError, match="outside"):
        completedp_betas(BETAS, CompletedPOptimConfig(base_batch_size=1, batch_size=20))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(base_batch_size=1), "base_batch_size"),
        (dict(train_tokens=1), "train_tokens"),
        (dict(base_batch_size=0, batch_size=1), "positive"),
    ],
)
def test_completedp_optim_config_validation(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        CompletedPOptimConfig(**kwargs)


def test_completedp_scheduler_keeps_ratios():
    model = _model()
    optim = AdamWConfig(lr=LR, completedp=CompletedPOptimConfig()).build(model)
    scheduler = CosWithWarmup(warmup=10, alpha_f=0.1)
    for step in (0, 5, 10, 50, 100):
        trainer = SimpleNamespace(global_step=step, max_steps=100)
        for group in optim.param_groups:
            scheduler.set_lr(group, trainer)  # type: ignore[arg-type]
        expected = scheduler.get_lr(LR, step, 100)
        for group in optim.param_groups:
            assert group["lr"] == pytest.approx(expected * group["lr_multiplier"])


def test_completedp_requires_completedp_model():
    with pytest.raises(OLMoConfigurationError, match="TransformerConfig.completedp"):
        AdamWConfig(completedp=CompletedPOptimConfig()).build(_model(completedp=False))


def test_completedp_model_requires_completedp_optimizer():
    with pytest.raises(OLMoConfigurationError, match="needs 'completedp' set"):
        AdamWConfig().build(_model())


def test_completedp_and_mup_are_exclusive():
    config = AdamWConfig(completedp=CompletedPOptimConfig(), mup=MuPConfig(base_width=BASE_WIDTH))
    with pytest.raises(OLMoConfigurationError, match="can't both be set"):
        config.build(_model())


def test_completedp_rejected_for_unsupported_optimizers():
    with pytest.raises(OLMoConfigurationError, match="not supported for LionConfig"):
        LionConfig(completedp=CompletedPOptimConfig()).build(_model())
    with pytest.raises(OLMoConfigurationError, match="not supported for MuonConfig"):
        MuonConfig().build(_model())
