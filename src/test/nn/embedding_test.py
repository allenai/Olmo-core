import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.embedding import SplitVocabEmbedding
from olmo_core.nn.transformer.config import TransformerConfig

BASE, EXTRA, D = 40, 6, 8


def test_shapes_and_total_vocab():
    emb = SplitVocabEmbedding(BASE, EXTRA, D)
    assert emb.weight.shape == (BASE, D)
    assert emb.extra_weight.shape == (EXTRA, D)
    assert emb.num_embeddings == BASE + EXTRA
    assert emb.full_weight().shape == (BASE + EXTRA, D)


def test_lookup_spans_both_blocks():
    emb = SplitVocabEmbedding(BASE, EXTRA, D)
    torch.nn.init.normal_(emb.weight)
    torch.nn.init.normal_(emb.extra_weight)
    ids = torch.tensor([[0, BASE - 1, BASE, BASE + EXTRA - 1]])
    out = emb(ids)
    torch.testing.assert_close(out[0, 0], emb.weight[0])
    torch.testing.assert_close(out[0, 1], emb.weight[BASE - 1])
    # IDs past the base block must resolve into the extra block, not go out of range.
    torch.testing.assert_close(out[0, 2], emb.extra_weight[0])
    torch.testing.assert_close(out[0, 3], emb.extra_weight[EXTRA - 1])


def test_requires_positive_extra_rows():
    with pytest.raises(ValueError, match="num_extra_embeddings"):
        SplitVocabEmbedding(BASE, 0, D)


@pytest.mark.parametrize("tied", [True, False])
def test_transformer_head_spans_base_vocab_only(tied: bool):
    """The whole point of naming the base block ``weight``: tying yields a base-width head."""
    cfg = TransformerConfig.olmo3_1M(
        vocab_size=BASE,
        n_extra_vocab=EXTRA,
        attn_backend=AttentionBackendName.torch,
        tie_word_embeddings=tied,
    )
    model = cfg.build(init_device="cpu")
    model.init_weights()

    assert isinstance(model.embeddings, SplitVocabEmbedding)
    assert model.lm_head.w_out.weight.shape[0] == BASE
    assert (model.lm_head.w_out.weight is model.embeddings.weight) is tied

    # Lookups accept the extra IDs even though they can never be predicted.
    logits = model(input_ids=torch.randint(0, BASE + EXTRA, (2, 5)))
    assert logits.shape == (2, 5, BASE)


@pytest.mark.parametrize("tied", [True, False])
def test_freezing_base_block_leaves_extra_rows_trainable(tied: bool):
    """mm_olmo's ``ft_embedding="lm_head"``: the pretrained rows are pinned, the added
    image-token rows keep learning. With a fused table this would be a partial-row freeze
    that ``requires_grad`` cannot express."""
    cfg = TransformerConfig.olmo3_1M(
        vocab_size=BASE,
        n_extra_vocab=EXTRA,
        attn_backend=AttentionBackendName.torch,
        tie_word_embeddings=tied,
    )
    model = cfg.build(init_device="cpu")
    model.init_weights()
    model.embeddings.weight.requires_grad_(False)

    extra_id = BASE + 2
    ids = torch.full((1, 4), extra_id)
    model(input_ids=ids).float().sum().backward()

    assert model.embeddings.weight.grad is None  # pinned (and, when tied, so is the head)
    assert model.embeddings.extra_weight.grad is not None
    assert model.embeddings.extra_weight.grad[2].abs().sum() > 0
