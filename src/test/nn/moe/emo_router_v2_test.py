import pytest
import torch

import olmo_core.ops.moe as ops
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.emo_router import EmoRouterV2
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


def test_segment_ids_from_eos() -> None:
    input_ids = torch.tensor([[1, 2, 0, 3, 0, 4], [0, 1, 0, 0, 2, 3]])
    assert torch.equal(
        ops.segment_ids_from_eos(input_ids, 0),
        torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 1, 2, 3, 3]]),
    )


@pytest.mark.parametrize("tokens", [[1, 2, 0, 3, 0, 4], [0, 1, 0, 0, 2, 0], [1, 2, 3], [0]])
def test_routing_segments_match_attention_documents(tokens):
    from olmo_core.data.utils import get_document_lengths

    ids = torch.tensor(tokens)
    lengths = get_document_lengths(ids, eos_token_id=0)
    expected = torch.arange(len(lengths)).repeat_interleave(lengths)
    torch.testing.assert_close(ops.segment_ids_from_eos(ids[None], 0)[0], expected)


def test_emo_config_validates_against_routed_experts() -> None:
    with pytest.raises(OLMoConfigurationError, match="number of routed experts"):
        EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=2,
            max_document_expert_pool=5,
        ).validate_for_router(num_experts=4, top_k=2)


def test_emo_router_uses_document_pool_across_all_routed_experts() -> None:
    router = MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=1,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=1,
            max_document_expert_pool=1,
            eval_document_expert_pool=1,
        ),
    ).build()
    assert isinstance(router, EmoRouterV2)
    router.eval()
    with torch.no_grad():
        router.weight.copy_(torch.eye(4).reshape(-1))

    x = torch.tensor([[[4.0, 0, 0, 0], [3.0, 0, 0, 0], [0, 0, 4.0, 0], [0, 0, 3.0, 0]]])
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    _, indices, counts, _ = router(x, False, segment_ids=segment_ids)

    assert indices is not None
    assert torch.equal(indices.squeeze(-1), torch.tensor([[0, 0, 2, 2]]))
    assert counts is not None and counts.shape == (4,)


def test_emo_router_requires_segment_ids() -> None:
    router = MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=1,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=1,
            max_document_expert_pool=2,
        ),
    ).build()
    with pytest.raises(OLMoConfigurationError, match="segment_ids"):
        router(torch.randn(1, 2, 4), False)


def test_emo_recompute_preserves_stochastic_routing_and_gradients():
    from copy import deepcopy

    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(118)
    original = (
        MoERouterConfigV2(
            d_model=8,
            num_experts=8,
            top_k=2,
            emo=EmoRouterConfig(
                eos_token_id=0, min_document_expert_pool=2, max_document_expert_pool=8
            ),
        )
        .build()
        .train()
    )
    reference, recomputed = deepcopy(original), deepcopy(original)
    tokens = torch.tensor([[1, 2, 0, 3, 4, 0, 5, 6]])
    segments = ops.segment_ids_from_eos(tokens, 0)
    source = torch.randn(1, 8, 8)
    results = []
    for router, enabled in ((reference, False), (recomputed, True)):
        x = source.clone().requires_grad_()
        selections = []

        def forward(x):
            weights, indices, _, _ = router(x, False, segment_ids=segments)
            selections.append(indices.detach().clone())
            return weights

        torch.manual_seed(209)
        output = (
            checkpoint(forward, x, use_reentrant=False, preserve_rng_state=True)
            if enabled
            else forward(x)
        )
        output.square().sum().backward()
        for selection in selections:
            torch.testing.assert_close(selection, selections[0], rtol=0, atol=0)
        results.append((output.detach(), x.grad, router.weight.grad, torch.get_rng_state()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def _brute_prefix_sums(per_token: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(per_token)
    for b in range(per_token.shape[0]):
        for t in range(per_token.shape[1]):
            same = (segment_ids[b, : t + 1] == segment_ids[b, t]).nonzero().squeeze(-1)
            out[b, t] = per_token[b, same].sum(dim=0)
    return out


@pytest.mark.parametrize(
    "segments",
    [[0] * 7, [0, 0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 1]],
)
def test_doc_prefix_sum_matches_per_document_cumsum(segments) -> None:
    torch.manual_seed(0)
    segment_ids = torch.tensor([segments, segments])
    per_token = torch.rand(2, len(segments), 5, dtype=torch.float64)
    prefix = ops.doc_prefix_sum(per_token, segment_ids)
    torch.testing.assert_close(prefix, _brute_prefix_sums(per_token, segment_ids))
    # The last token of each document sees its whole document's totals.
    whole = ops.doc_sum_scatter(per_token, segment_ids)
    last = torch.ones_like(segment_ids, dtype=torch.bool)
    last[:, :-1] = segment_ids[:, 1:] != segment_ids[:, :-1]
    torch.testing.assert_close(prefix[last], whole[last])


def _emo_keep_mask(
    monkeypatch, logits: torch.Tensor, segment_ids: torch.Tensor, *, causal: bool, seed: int = 5
) -> torch.Tensor:
    """The EMO pool mask a router with pools in [2, 8] of 8 experts builds for ``logits``.

    Mirrors MaxText's ``EmoEvalPoolsTest._mask``: the router weight is the identity, so its
    expert logits are ``logits``, and pool sizes are sampled with ``seed``.
    """
    router = MoERouterConfigV2(
        d_model=8,
        num_experts=8,
        top_k=2,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=2,
            max_document_expert_pool=8,
            causal_pool=causal,
        ),
    ).build()
    assert isinstance(router, EmoRouterV2)
    with torch.no_grad():
        router.weight.copy_(torch.eye(8).reshape(-1))
    masks = []
    pool_keep_mask = ops.pool_keep_mask

    def capture(document_scores, pool_sizes):
        keep = pool_keep_mask(document_scores, pool_sizes)
        masks.append(keep)
        return keep

    monkeypatch.setattr(ops, "pool_keep_mask", capture)
    torch.manual_seed(seed)
    router(logits, False, segment_ids=segment_ids)
    (keep,) = masks
    return keep


# Two documents (6 + 10 tokens) in row 0, one 16-token document in row 1, as in MaxText's tests.
# Seed 5 draws pools of 3, 7 and 6 experts for them, so every pool restricts its document.
_SEGMENTS = torch.tensor([[0] * 6 + [1] * 10, [0] * 16])


def test_causal_pool_ignores_later_tokens(monkeypatch) -> None:
    """With ``causal_pool`` a token's pool depends only on its document's tokens up to itself."""
    logits = torch.randn(2, 16, 8, generator=torch.Generator().manual_seed(0))
    later = logits.clone()
    later[:, 9:] += 5.0 * torch.randn(2, 7, 8, generator=torch.Generator().manual_seed(2))
    keep = _emo_keep_mask(monkeypatch, logits, _SEGMENTS, causal=True)
    keep_later = _emo_keep_mask(monkeypatch, later, _SEGMENTS, causal=True)
    assert torch.equal(keep[:, :9], keep_later[:, :9])
    assert not torch.equal(keep[:, 9:], keep_later[:, 9:]), "later pools should follow the changes"


def test_causal_pool_first_token_keeps_its_own_top_experts(monkeypatch) -> None:
    """A document's first token keeps its own top experts; its last token's pool is the whole
    document's."""
    logits = torch.randn(2, 16, 8, generator=torch.Generator().manual_seed(0))
    causal = _emo_keep_mask(monkeypatch, logits, _SEGMENTS, causal=True)
    whole = _emo_keep_mask(monkeypatch, logits, _SEGMENTS, causal=False)
    for b, first, last in ((0, 0, 5), (0, 6, 15), (1, 0, 15)):
        kept = causal[b, first]
        assert 0 < int(kept.sum()) < 8, "pick a seed whose pool restricts the first token"
        assert logits[b, first][kept].min() >= logits[b, first][~kept].max()
        assert torch.equal(causal[b, last], whole[b, last])


def test_multimodal_emo_segments_follow_example_ids() -> None:
    from olmo_core.nn.vision.multimodal import segment_ids_from_example_ids

    example_ids = torch.tensor([[0, 0, 1, 1, 1, 2, -1, -1], [3, 3, 3, 3, 3, 3, 3, 3]])
    assert torch.equal(
        segment_ids_from_example_ids(example_ids),
        torch.tensor([[0, 0, 1, 1, 1, 2, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0]]),
    )


def test_doc_prefix_sum_is_exact_over_long_rows() -> None:
    """Late in a long row, the prefix sums (and the pools they pick) match exact per-document
    sums: the whole-row running sum is accumulated in float64, not float32."""
    gen = torch.Generator().manual_seed(0)
    seq_len, num_experts = 262_144, 32
    scores = torch.randn(1, seq_len, num_experts, generator=gen).softmax(dim=-1)
    # Documents of 1-4096 tokens, so many start deep into the row.
    lengths = torch.randint(1, 4097, (seq_len,), generator=gen)
    boundaries = lengths.cumsum(0)
    boundaries = boundaries[boundaries < seq_len]
    doc_of = torch.zeros(seq_len, dtype=torch.long)
    doc_of[boundaries] = 1
    segment_ids = doc_of.cumsum(0).unsqueeze(0)

    exact = torch.empty_like(scores, dtype=torch.float64)
    starts = [0, *boundaries.tolist(), seq_len]
    for a, b in zip(starts[:-1], starts[1:]):
        exact[0, a:b] = scores[0, a:b].double().cumsum(dim=0)

    prefix = ops.doc_prefix_sum(scores, segment_ids)
    assert prefix.dtype == scores.dtype
    torch.testing.assert_close(prefix.double(), exact, rtol=1e-6, atol=1e-6)
    pool = torch.full_like(segment_ids, 8)
    keep, keep_exact = ops.pool_keep_mask(prefix, pool), ops.pool_keep_mask(exact.float(), pool)
    assert (keep != keep_exact).any(dim=-1).float().mean() < 1e-4
