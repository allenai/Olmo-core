"""Mechanisms of the four-node multimodal SFT sweep: image loss share, rebalanced mix, high res."""

import pytest
import torch

from olmo_core.data.multimodal.mixtures import olmo35_sft_mm_v1 as v1
from olmo_core.internal.olmo35_mm_sft import image_loss_scale
from olmo_core.train.train_module.transformer.multimodal_train_module import _scale_source_losses


@pytest.mark.parametrize("share", [0.05, 0.10, 0.25])
def test_image_loss_scale_gives_the_requested_share(share):
    text_mass, image_mass = 1.7e10, 2.53e8
    w = image_loss_scale(text_mass, image_mass, share)
    assert w * image_mass / (text_mass + w * image_mass) == pytest.approx(share)


def test_image_loss_scale_rejects_bad_shares():
    with pytest.raises(Exception):
        image_loss_scale(1.0, 1.0, 1.0)


def test_scale_source_losses_touches_only_the_listed_sources():
    loss_masks = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    example_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]])
    batch = {
        "loss_masks": loss_masks,
        "example_ids": example_ids,
        "pack_source_names": [["text_sft", "doc_qa"], ["text_sft"]],
    }
    out = _scale_source_losses(batch, {"doc_qa": 7.0})
    expected = torch.tensor([[1.0, 1.0, 3.5, 3.5, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    assert torch.equal(out["loss_masks"], expected)
    assert torch.equal(batch["loss_masks"], loss_masks)  # the input batch is not modified
    assert out["pack_source_names"] is batch["pack_source_names"]


def test_rebalance_keeps_the_dose_and_moves_tokens_to_the_targets():
    base = v1.expected_examples()
    new = v1.rebalanced_examples()
    tokens = lambda ex: sum(ex[n] * v1.MEAN_TOKENS[n] for n in ex)  # noqa: E731
    assert tokens(new) == pytest.approx(tokens(base), rel=1e-9)
    for name, rows in v1.REBALANCE_CUT_ROWS.items():
        assert new[name] == pytest.approx(v1.REBALANCE_ANCHOR_EPOCHS * rows)
        assert new[name] < base[name]
    ratios = {new[n] / base[n] for n in v1.REBALANCE_TARGETS}
    assert len({round(r, 9) for r in ratios}) == 1 and next(iter(ratios)) > 1.0
    untouched = set(base) - set(v1.REBALANCE_CUT) - set(v1.REBALANCE_TARGETS)
    assert untouched and all(new[n] == pytest.approx(base[n]) for n in untouched)


def test_rebalance_targets_are_multi_image_and_document_sources():
    targets = set(v1.REBALANCE_TARGETS)
    assert {"doc_qa", "info_qa", "chart_qa_weighted", "correction_qa_multi_only_max5"} <= targets
    assert any(n.startswith("mantis_instruct_") for n in targets)
    assert any(n.startswith("cosyn_multidoc_") for n in targets)
    assert not targets & set(v1.REBALANCE_CUT)


def test_high_res_sources_get_the_draw_and_keep_the_dose():
    from olmo_core.internal.olmo35_mm_sft import stage2_v1_image_sources

    hires = {"doc_qa": 2500.0, "info_qa": 2600.0}
    sources = stage2_v1_image_sources(
        3.5e9, 65_536, high_res_sources=list(hires), high_res_p=1.0, high_res_mean_tokens=hires
    )
    assert sources["doc_qa"].source.p_high_res == 1.0
    assert getattr(sources["text_vqa"].source, "p_high_res", 0.0) == 0.0
    total = sum(
        s.loss_share / v1.MEAN_LOSS_WEIGHTS[n] * float(s.mean_tokens) for n, s in sources.items()
    )
    assert total == pytest.approx(3.5e9, rel=1e-6)
    with pytest.raises(Exception):
        stage2_v1_image_sources(3.5e9, 65_536, high_res_sources=["doc_qa"], high_res_p=0.5)
    # At p=1.0 the measured high-resolution means are the default.
    measured = stage2_v1_image_sources(3.5e9, 65_536, high_res_sources=["doc_qa"])
    assert measured["doc_qa"].mean_tokens == v1.HIGH_RES_MEAN_TOKENS["doc_qa"]
