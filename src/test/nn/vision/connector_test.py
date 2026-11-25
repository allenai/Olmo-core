import pytest
import torch

from olmo_core.nn.vision import (
    ImagePoolingType,
    ImageProjectorType,
    VisionConnector,
    VisionConnectorConfig,
    VisionEncoderConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_cfg(
    pooling_type: ImagePoolingType = ImagePoolingType.attention_meanq,
    projector_type: ImageProjectorType = ImageProjectorType.mlp,
    num_input_layers: int = 1,
    output_dim: int = 128,
    pooling_attention_mask: bool = False,
) -> VisionConnectorConfig:
    return VisionConnectorConfig(
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_head_dim=16,
        output_dim=output_dim,
        num_input_layers=num_input_layers,
        pooling_type=pooling_type,
        pooling_attention_mask=pooling_attention_mask,
        projector_type=projector_type,
        mlp_hidden_size=64,
    )


def _features(
    batch: int, n_total_patches: int, num_layers: int = 1, emb_dim: int = 64
) -> torch.Tensor:
    return torch.randn(batch, n_total_patches, num_layers * emb_dim)


def _row_major_pooled_idx(
    batch: int,
    grid_h: int,
    grid_w: int,
    pool_h: int = 2,
    pool_w: int = 2,
) -> torch.Tensor:
    """Build a ``pooled_patches_idx`` for a single-crop ``grid_h × grid_w`` grid.

    Groups patches into ``pool_h × pool_w`` non-overlapping windows in
    row-major order. No padding (``-1``) entries.
    """
    assert grid_h % pool_h == 0 and grid_w % pool_w == 0
    nH, nW = grid_h // pool_h, grid_w // pool_w
    pool_size = pool_h * pool_w
    idx = torch.zeros(nH * nW, pool_size, dtype=torch.long)
    for i in range(nH):
        for j in range(nW):
            group = i * nW + j
            slot = 0
            for di in range(pool_h):
                for dj in range(pool_w):
                    patch_row = i * pool_h + di
                    patch_col = j * pool_w + dj
                    idx[group, slot] = patch_row * grid_w + patch_col
                    slot += 1
    return idx.unsqueeze(0).expand(batch, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# VisionConnectorConfig
# ---------------------------------------------------------------------------


def test_from_vision_encoder():
    vis_cfg = VisionEncoderConfig(
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_head_dim=16,
    )
    conn_cfg = VisionConnectorConfig.from_vision_encoder(vis_cfg, output_dim=256)
    assert conn_cfg.image_emb_dim == 64
    assert conn_cfg.output_dim == 256
    assert conn_cfg.num_input_layers == 1


def test_pooling_input_dim_single_layer():
    cfg = _small_cfg()
    assert cfg.pooling_input_dim == 64  # 1 * image_emb_dim


def test_pooling_input_dim_multi_layer():
    cfg = _small_cfg(num_input_layers=2)
    assert cfg.pooling_input_dim == 128  # 2 * image_emb_dim


def test_projector_input_dim_after_pooling():
    cfg = _small_cfg(pooling_type=ImagePoolingType.attention_meanq)
    assert cfg.projector_input_dim == 64  # image_emb_dim after pooling


def test_projector_input_dim_no_pooling():
    cfg = _small_cfg(pooling_type=ImagePoolingType.none)
    assert cfg.projector_input_dim == 64  # equals pooling_input_dim when no pooling


def test_build_returns_vision_connector():
    cfg = _small_cfg()
    conn = cfg.build(init_device="cpu")
    assert isinstance(conn, VisionConnector)


# ---------------------------------------------------------------------------
# VisionConnector — attention_meanq pooling + mlp projector
# ---------------------------------------------------------------------------


class TestConnectorAttentionMLP:
    def setup_method(self):
        self.cfg = _small_cfg()
        self.conn = self.cfg.build(init_device="cpu")

    def test_output_shape_4x_patch_reduction(self):
        # 4×4 = 16 patches → 2×2 pool = 4 pooled tokens with pool_size=4
        x = _features(2, 16)
        idx = _row_major_pooled_idx(2, 4, 4)
        out = self.conn(x, idx)
        assert out.shape == (2, 4, self.cfg.output_dim)

    def test_output_shape_single_patch_grid(self):
        # 2×2 grid → 1 pooled token
        x = _features(1, 4)
        idx = _row_major_pooled_idx(1, 2, 2)
        out = self.conn(x, idx)
        assert out.shape == (1, 1, self.cfg.output_dim)

    def test_output_dtype_matches_input(self):
        x = _features(1, 16).to(torch.float32)
        idx = _row_major_pooled_idx(1, 4, 4)
        out = self.conn(x, idx)
        assert out.dtype == torch.float32

    def test_pooling_module_present(self):
        assert self.conn.pooling is not None

    @pytest.mark.parametrize("batch", [1, 3])
    def test_various_batch_sizes(self, batch: int):
        x = _features(batch, 16)
        idx = _row_major_pooled_idx(batch, 4, 4)
        out = self.conn(x, idx)
        assert out.shape == (batch, 4, self.cfg.output_dim)

    def test_deterministic_eval(self):
        x = _features(2, 16)
        idx = _row_major_pooled_idx(2, 4, 4)
        self.conn.eval()
        out1 = self.conn(x, idx)
        out2 = self.conn(x, idx)
        assert torch.allclose(out1, out2)

    def test_meta_device(self):
        conn = _small_cfg().build(init_device="meta")
        assert next(iter(conn.parameters())).device.type == "meta"


# ---------------------------------------------------------------------------
# VisionConnector — multi-crop layout
# ---------------------------------------------------------------------------


def test_multi_crop_via_flat_patch_index():
    """Two crops worth of patches concatenated, single set of pool groups."""
    cfg = _small_cfg()
    conn = cfg.build(init_device="cpu")
    # 2 crops × 16 patches = 32 patches total; pool into 8 groups of 4.
    x = _features(1, 32)
    # Two stacked 4×4 row-major grids: first crop is patches 0..15, second 16..31.
    idx_crop0 = _row_major_pooled_idx(1, 4, 4)  # (1, 4, 4) indices 0..15
    idx_crop1 = _row_major_pooled_idx(1, 4, 4) + 16  # shift to second crop
    idx = torch.cat([idx_crop0, idx_crop1], dim=1)  # (1, 8, 4)
    out = conn(x, idx)
    assert out.shape == (1, 8, cfg.output_dim)


# ---------------------------------------------------------------------------
# VisionConnector — padding via -1 indices
# ---------------------------------------------------------------------------


def test_padding_indices_zero_out_features():
    """Pool groups with -1 entries should ignore those slots when masked."""
    cfg = _small_cfg(pooling_attention_mask=True)
    conn = cfg.build(init_device="cpu")
    x = _features(1, 4)
    # One group with two real patches and two padded entries.
    idx = torch.tensor([[[0, 1, -1, -1]]], dtype=torch.long)
    out = conn(x, idx)
    assert out.shape == (1, 1, cfg.output_dim)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# VisionConnector — multi-layer input
# ---------------------------------------------------------------------------


def test_multi_layer_input():
    cfg = _small_cfg(num_input_layers=2)
    conn = cfg.build(init_device="cpu")
    x = _features(2, 16, num_layers=2)  # (2, 16, 128)
    idx = _row_major_pooled_idx(2, 4, 4)
    out = conn(x, idx)
    assert out.shape == (2, 4, cfg.output_dim)


# ---------------------------------------------------------------------------
# VisionConnector — no pooling variant (pool_size=1)
# ---------------------------------------------------------------------------


class TestConnectorNoPooling:
    def setup_method(self):
        self.cfg = _small_cfg(pooling_type=ImagePoolingType.none)
        self.conn = self.cfg.build(init_device="cpu")

    def test_no_pooling_module(self):
        assert self.conn.pooling is None

    def test_output_shape_unchanged_patch_count(self):
        # With pool_size=1, each patch becomes its own pooled token.
        x = _features(2, 16)
        idx = torch.arange(16, dtype=torch.long).view(1, 16, 1).expand(2, -1, -1).contiguous()
        out = self.conn(x, idx)
        assert out.shape == (2, 16, self.cfg.output_dim)


# ---------------------------------------------------------------------------
# VisionConnector — linear projector
# ---------------------------------------------------------------------------


def test_linear_projector():
    cfg = _small_cfg(projector_type=ImageProjectorType.linear, pooling_type=ImagePoolingType.none)
    conn = cfg.build(init_device="cpu")
    x = _features(2, 16)
    idx = torch.arange(16, dtype=torch.long).view(1, 16, 1).expand(2, -1, -1).contiguous()
    out = conn(x, idx)
    assert out.shape == (2, 16, cfg.output_dim)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_config_round_trip():
    from olmo_core.config import Config

    cfg = _small_cfg()
    assert isinstance(cfg, Config)
    d = cfg.as_config_dict()
    assert d["pooling_type"] == "attention_meanq"
    assert d["projector_type"] == "mlp"


# ---------------------------------------------------------------------------
# Pooling attention and patch gather: the generic SDPA and advanced-indexing results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("masked", [False, True])
def test_few_key_attention_matches_sdpa(masked: bool):
    from olmo_core.nn.vision.connector import _few_key_attention

    torch.manual_seed(0)
    G, P, H, D = 12, 4, 3, 8
    q = torch.randn(G, H, D, requires_grad=True)
    k = torch.randn(G, P, H, D, requires_grad=True)
    v = torch.randn(G, P, H, D, requires_grad=True)
    valid = None
    if masked:
        valid = torch.rand(G, P) < 0.6
        valid[0] = False  # a group without any valid key
        valid[1] = True
    out = _few_key_attention(q, k, v, valid)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, None],
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=None if valid is None else valid[:, None, None, :],
    )[:, :, 0]
    torch.testing.assert_close(out, ref)
    for a, b in zip(
        torch.autograd.grad(out.square().sum(), [q, k, v]),
        torch.autograd.grad(ref.square().sum(), [q, k, v]),
    ):
        torch.testing.assert_close(a, b)


def _reference_connector(connector: VisionConnector, features, idx) -> torch.Tensor:
    """The connector with advanced-indexing gather and generic SDPA pooling."""
    pool, cfg = connector.pooling, connector.cfg
    assert pool is not None
    B, n_pooled, pool_size = idx.shape
    valid = idx >= 0
    batch_idx = torch.arange(B)[:, None, None].expand_as(idx)
    flat = (features[batch_idx, idx.clamp(min=0)] * valid[..., None]).reshape(
        B * n_pooled, pool_size, -1
    )
    attn_mask = None
    if cfg.pooling_attention_mask:
        attn_mask = valid.reshape(B * n_pooled, 1, 1, pool_size)
        denom = valid.reshape(B * n_pooled, pool_size).sum(-1).clamp(min=1)
        query = flat.sum(dim=1, keepdim=True) / denom[:, None, None]
    else:
        query = flat.mean(dim=1, keepdim=True)
    heads = (pool.num_heads, pool.head_dim)
    q = pool.wq(query).reshape(B * n_pooled, 1, *heads).transpose(1, 2)
    k = pool.wk(flat).reshape(B * n_pooled, pool_size, *heads).transpose(1, 2)
    v = pool.wv(flat).reshape(B * n_pooled, pool_size, *heads).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    pooled = pool.wo(out.transpose(1, 2).reshape(B * n_pooled, 1, -1))
    return connector.projector(pooled.reshape(B, n_pooled, -1))


@pytest.mark.parametrize("pooling_attention_mask", [False, True])
def test_connector_matches_the_generic_gather_and_attention(pooling_attention_mask: bool):
    torch.manual_seed(0)
    connector = _small_cfg(pooling_attention_mask=pooling_attention_mask).build()
    connector.reset_parameters()
    features = _features(2, 3 * 16).requires_grad_()
    idx = torch.cat([_row_major_pooled_idx(2, 4, 4), _row_major_pooled_idx(2, 4, 4) + 16], dim=1)
    idx[0, 1, 2:] = -1  # a partly padded group
    idx[1, -3:] = -1  # padded rows
    params = [features, *connector.parameters()]
    out = connector(features, idx)
    ref = _reference_connector(connector, features, idx)
    torch.testing.assert_close(out, ref)
    for a, b in zip(
        torch.autograd.grad(out.square().sum(), params),
        torch.autograd.grad(ref.square().sum(), params),
    ):
        torch.testing.assert_close(a, b)
