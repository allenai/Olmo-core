from types import SimpleNamespace

import pytest
import torch

from olmo_core.nn.attention import flash_attn_api


@pytest.mark.parametrize("separate_metadata", [False, True])
def test_flash4_varlen_metadata_is_keyword_bound(monkeypatch, separate_metadata):
    """FA4 may insert optional qv before the sequence metadata arguments."""
    q = torch.randn(1, 7, 2, 8)
    cu_q = torch.tensor([0, 3, 7], dtype=torch.int32)
    cu_k = cu_q.clone() if separate_metadata else cu_q

    def varlen(
        q,
        k,
        v,
        qv=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        **kwargs,
    ):
        assert qv is None
        assert cu_seqlens_q is cu_q
        assert cu_seqlens_k is cu_k
        assert max_seqlen_q == 4
        assert max_seqlen_k == 4
        assert q.shape == (7, 2, 8)
        assert kwargs["causal"]
        return q, None

    monkeypatch.setattr(
        flash_attn_api, "flash_attn_4", SimpleNamespace(flash_attn_varlen_func=varlen)
    )
    metadata = (
        dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4, max_seqlen_k=4)
        if separate_metadata
        else dict(cu_seqlens=cu_q, max_seqlen=4)
    )
    result = flash_attn_api.dispatch_flash_attn_4(q, q, q, causal=True, **metadata)
    torch.testing.assert_close(result, q.flatten(0, 1))
