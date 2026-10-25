"""
Document mode of :class:`~olmo_core.nn.vision.MultimodalLM`: language models with recurrent
sequence mixers (KDA) cannot take attention masks, so packed examples are isolated through
document boundaries derived from ``example_ids`` and no masks or positions are sent. The mask
path stays in place for attention-only language models.
"""

from unittest.mock import Mock

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.kda import KimiDeltaAttentionConfig
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    MultimodalOLMoDDPModel,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.nn.vision.multimodal import (
    document_lengths_from_example_ids,
    has_sibling_branches,
)

_D_MODEL = 16
_VOCAB = 64


def _lm_config(mixer) -> OLMoDDPModelConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)
    block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=mixer,
        layer_norm=layer_norm,
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=_D_MODEL, hidden_size=32, num_experts=1, bias=False, dtype=DType.float32
        ),
        shared_experts_router=None,
    )
    return OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=_D_MODEL,
        vocab_size=_VOCAB,
        n_layers=2,
        lm_head=LMHeadConfig(bias=False, dtype=DType.float32),
        block=block,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )


def _attention_mixer() -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=2,
        n_kv_heads=2,
        bias=False,
        backend=AttentionBackendName.torch,
        dtype=DType.float32,
    )


def _multimodal_config(mixer) -> MultimodalLMConfig:
    vision = VisionEncoderConfig(
        name=VisionEncoderType.openai,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=5,
        image_norm_eps=1e-5,
    )
    return MultimodalLMConfig(
        lm=_lm_config(mixer),
        vision=vision,
        connector=VisionConnectorConfig.from_vision_encoder(vision, output_dim=_D_MODEL),
        image_patch_token_id=1,
    )


def _attention_model(init_device: str = "cpu") -> MultimodalOLMoDDPModel:
    torch.manual_seed(0)
    model = MultimodalOLMoDDPModel(_multimodal_config(_attention_mixer()), init_device=init_device)
    model.init_weights(max_seq_len=64, device=torch.device(init_device))
    model.eval()
    return model


def test_document_lengths_follow_contiguous_example_runs():
    example_ids = torch.tensor([[0, 0, 0, 1, 1, -1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0]])
    doc_lens, max_doc_lens = document_lengths_from_example_ids(example_ids)
    assert doc_lens.tolist() == [[3, 2, 3], [8, 0, 0]]
    assert doc_lens.dtype == torch.int32
    assert max_doc_lens == [3, 8]
    assert doc_lens.sum(dim=1).tolist() == [8, 8]


def test_attention_only_language_models_keep_the_mask_path():
    model = _attention_model()
    assert not model.uses_document_boundaries()
    lm_forward = Mock(wraps=model.lm.forward)
    model.lm.forward = lm_forward  # type: ignore[method-assign]
    input_ids = torch.randint(2, _VOCAB, (1, 8))
    example_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        model(input_ids, example_ids=example_ids)
    kwargs = lm_forward.call_args.kwargs
    assert kwargs["and_mask"] is not None and kwargs["and_mask"].shape == (1, 1, 8, 8)
    assert "doc_lens" not in kwargs and "max_doc_lens" not in kwargs


def test_recurrent_language_models_are_detected():
    kda = KimiDeltaAttentionConfig(n_heads=2, head_dim=8, dtype=DType.float32)
    model = MultimodalOLMoDDPModel(_multimodal_config(kda), init_device="meta")
    assert model.uses_document_boundaries()


def test_document_mode_sends_boundaries_and_no_masks_or_positions():
    # The torch attention backend takes no document boundaries, so only the LM call is checked.
    model = _attention_model()
    model._document_mode = True
    lm_forward = Mock(return_value=torch.zeros(2, 8, _VOCAB))
    model.lm.forward = lm_forward  # type: ignore[method-assign]
    input_ids = torch.randint(2, _VOCAB, (2, 8))
    example_ids = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1], [0, 0, 0, 0, 0, 0, 0, 0]])
    position_ids = torch.arange(8).repeat(2, 1)
    with torch.no_grad():
        model(input_ids, example_ids=example_ids, position_ids=position_ids)
    kwargs = lm_forward.call_args.kwargs
    assert kwargs["and_mask"] is None and kwargs["or_mask"] is None
    assert kwargs["flex_attn_block_mask"] is None and kwargs["position_ids"] is None
    assert kwargs["doc_lens"].tolist() == [[3, 3, 2], [8, 0, 0]]
    assert kwargs["max_doc_lens"] == [3, 8]


def test_document_mode_treats_an_unpacked_batch_as_one_document_per_row():
    model = _attention_model()
    model._document_mode = True
    lm_forward = Mock(return_value=torch.zeros(3, 6, _VOCAB))
    model.lm.forward = lm_forward  # type: ignore[method-assign]
    with torch.no_grad():
        model(torch.randint(2, _VOCAB, (3, 6)))
    kwargs = lm_forward.call_args.kwargs
    assert kwargs["doc_lens"].tolist() == [[6], [6], [6]] and kwargs["max_doc_lens"] == [6, 6, 6]


@pytest.mark.gpu
def test_document_mode_isolates_packed_examples_through_kda():
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("KDA kernels need a GPU")
    torch.manual_seed(0)
    kda = KimiDeltaAttentionConfig(n_heads=2, head_dim=8, dtype=DType.float32)
    model = MultimodalOLMoDDPModel(_multimodal_config(kda), init_device="cuda")
    model.init_weights(max_seq_len=64, device=torch.device("cuda"))
    model.eval()
    assert model.uses_document_boundaries()
    a = torch.randint(2, _VOCAB, (1, 40), device="cuda")
    b = torch.randint(2, _VOCAB, (1, 24), device="cuda")
    packed = torch.cat([a, b], dim=1)
    example_ids = torch.tensor([[0] * 40 + [1] * 24], device="cuda")
    with torch.no_grad():
        alone = model(b)
        packed_logits = model(packed, example_ids=example_ids)[:, 40:]
        leaked = model(packed)[:, 40:]  # one document: state carries over from example A
    torch.testing.assert_close(packed_logits, alone, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(leaked, alone, rtol=1e-3, atol=1e-3)


def test_document_mode_accepts_packer_subsegment_ids_without_branches():
    # The packer always emits ``subsegment_ids``: a constant id per branch-free example (and the
    # ATTEND_ALL prefix id plus one branch id for a single-annotation example), which carries
    # nothing beyond the example boundaries; the first bridge smoke tripped on exactly this.
    assert not has_sibling_branches(
        torch.tensor([[0, 0, 0, 1, 1, 1]]), torch.tensor([[0, 0, 0, 1, 1, 1]])
    )
    assert not has_sibling_branches(torch.tensor([[10000, 10000, 5, 5, 5, 5]]))
    assert has_sibling_branches(torch.tensor([[10000, 10000, 1, 1, 2, 2]]))
    assert has_sibling_branches(
        torch.tensor([[0, 0, 1, 1, 7, 7]]), torch.tensor([[0, 0, 0, 0, 1, 1]])
    )
    model = _attention_model()
    model._document_mode = True
    lm_forward = Mock(return_value=torch.zeros(1, 6, _VOCAB))
    model.lm.forward = lm_forward  # type: ignore[method-assign]
    input_ids = torch.randint(2, _VOCAB, (1, 6))
    with torch.no_grad():
        model(
            input_ids,
            subsegment_ids=torch.tensor([[0, 0, 0, 1, 1, 1]]),
            example_ids=torch.tensor([[0, 0, 0, 1, 1, 1]]),
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),  # the packer always sends these
        )
    kwargs = lm_forward.call_args.kwargs
    assert kwargs["doc_lens"].tolist() == [[3, 3]] and kwargs["and_mask"] is None
    assert kwargs["position_ids"] is None


def test_document_mode_rejects_sibling_branch_packing():
    model = _attention_model()
    model._document_mode = True
    input_ids = torch.randint(2, _VOCAB, (1, 6))
    with pytest.raises(ValueError, match="Sibling-branch packing"):
        model(
            input_ids,
            subsegment_ids=torch.tensor([[0, 0, 1, 1, 2, 2]]),
            position_ids=torch.tensor([[0, 1, 2, 3, 2, 3]]),
        )
