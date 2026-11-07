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


def _causal_router(pool: int, num_experts: int = 8, top_k: int = 2) -> EmoRouterV2:
    router = MoERouterConfigV2(
        d_model=num_experts,
        num_experts=num_experts,
        top_k=top_k,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=pool,
            max_document_expert_pool=pool,
            causal_pool=True,
        ),
    ).build()
    assert isinstance(router, EmoRouterV2)
    with torch.no_grad():
        router.weight.copy_(torch.eye(num_experts).reshape(-1))
    return router


def test_causal_pool_ignores_later_tokens() -> None:
    torch.manual_seed(0)
    router = _causal_router(pool=3)
    segment_ids = torch.tensor([[0] * 6 + [1] * 10])
    x = torch.randn(1, 16, 8)
    later = x.clone()
    later[:, 9:] += 5.0 * torch.randn(1, 7, 8)
    _, idx, _, _ = router(x, False, segment_ids=segment_ids)
    _, idx_later, _, _ = router(later, False, segment_ids=segment_ids)
    assert idx is not None and idx_later is not None
    assert torch.equal(idx[:, :9], idx_later[:, :9])
    assert not torch.equal(idx[:, 9:], idx_later[:, 9:])


def test_causal_pool_first_token_unconstrained_last_token_whole_document() -> None:
    torch.manual_seed(0)
    segment_ids = torch.tensor([[0] * 6 + [1] * 10])
    x = torch.randn(1, 16, 8)
    causal = _causal_router(pool=2)
    whole = _causal_router(pool=2)
    whole.emo.causal_pool = False
    free = MoERouterConfigV2(d_model=8, num_experts=8, top_k=2).build()
    with torch.no_grad():
        free.weight.copy_(torch.eye(8).reshape(-1))
    _, idx_causal, _, _ = causal(x, False, segment_ids=segment_ids)
    _, idx_whole, _, _ = whole(x, False, segment_ids=segment_ids)
    _, idx_free, _, _ = free(x, False)
    assert idx_causal is not None and idx_whole is not None and idx_free is not None
    for first, last in ((0, 5), (6, 15)):
        # pool == top_k: the first token's pool is exactly its own top-k experts.
        assert set(idx_causal[0, first].tolist()) == set(idx_free[0, first].tolist())
        assert set(idx_causal[0, last].tolist()) == set(idx_whole[0, last].tolist())


def test_multimodal_emo_segments_follow_example_ids() -> None:
    from olmo_core.nn.vision.multimodal import segment_ids_from_example_ids

    example_ids = torch.tensor([[0, 0, 1, 1, 1, 2, -1, -1], [3, 3, 3, 3, 3, 3, 3, 3]])
    assert torch.equal(
        segment_ids_from_example_ids(example_ids),
        torch.tensor([[0, 0, 1, 1, 1, 2, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0]]),
    )
