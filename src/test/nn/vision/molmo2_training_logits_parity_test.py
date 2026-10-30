"""Training-path parity / correctness for Molmo2 :class:`MultimodalLM`.

Two layers of validation for "stage 1" training:

1. **Training-loss parity vs HF** (GPU, requires a cached Molmo2 checkpoint):
   feed an identical image + prompt + response example through HF Molmo2 and our
   :class:`MultimodalLM`, then check that (a) the per-token logits match across
   *all* response positions and (b) the float-``loss_masks``-weighted cross entropy
   (mm_olmo's reduction) computed from each set of logits agrees.

2. **Subsegment branch isolation** (CPU, tiny random model): the packed
   multi-annotation attention must isolate sibling branches — changing one branch's
   tokens must not change another branch's logits — while still depending on the
   shared prefix. Also checks the float ``root_subsegments`` loss weighting flows end
   to end. This is HF-independent and exercises the new ``and_mask`` / ``position_ids``
   plumbing directly.
"""

import numpy as np
import pytest
import torch

from olmo_core.data.multimodal import build_packed_sequence
from olmo_core.data.multimodal.sequence_builder import ATTEND_ALL_SUBSEGMENT_ID
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.testing import requires_gpu

from .molmo2_logits_parity_test import MOLMO2_VARIANTS, _build_ours, _hf_cache_has
from .multimodal_test import (
    _IMAGE_PATCH_TOKEN,
    _LM_VOCAB,
    _make_inputs,
    _tiny_multimodal_cfg,
)

# ---------------------------------------------------------------------------
# 1. Training-loss parity vs HF (single caption branch)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("model_id", MOLMO2_VARIANTS)
def test_molmo2_training_loss_parity(model_id: str):
    """Logits (all response positions) and the float-weighted loss match HF.

    .. note::
        Molmo2-O-7B uses per-layer YaRN attention scaling not yet implemented in
        :class:`MultimodalLM`; skipped (same as the inference parity tests).
    """
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in HF cache")
    if "O-7B" in model_id:
        pytest.skip(f"{model_id} uses per-layer YaRN attention scaling; training parity skipped.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hf, ours, cfg = _build_ours(model_id, device, dtype)
    hf = hf.to(device=device, dtype=dtype).eval()

    torch.manual_seed(0)
    patch_size = cfg.vision.image_patch_size
    n_patches = cfg.vision.image_num_pos

    pixel_values_hf = torch.randn(
        1, n_patches, 3 * patch_size * patch_size, dtype=dtype, device=device
    )
    image_token_pooling = torch.arange(4, dtype=torch.long, device=device).view(1, 4)
    image_grids = torch.tensor([[1, 1, 0, 0]], dtype=torch.long, device=device)
    image_num_crops = torch.tensor([1], dtype=torch.long, device=device)

    image_end_id = hf.config.image_end_token_id
    image_patch_id = hf.config.image_patch_id
    base_vocab = cfg.lm.vocab_size - 128
    prompt = torch.randint(0, base_vocab, (3,), device=device).tolist()
    response = torch.randint(0, base_vocab, (5,), device=device).tolist()
    seq = [image_end_id, image_patch_id, image_end_id] + prompt + response
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)

    # Loss only on the response tokens (binary float weights, root_subsegments at 1 branch).
    n_prefix = 3 + len(prompt)
    loss_masks = torch.zeros(1, len(seq), device=device)
    loss_masks[0, n_prefix:] = 1.0
    # Next-token aligned labels (predict token i+1 from position i).
    labels = torch.full((1, len(seq)), -100, dtype=torch.long, device=device)
    labels[0, :-1] = input_ids[0, 1:]
    labels[0, ~(loss_masks[0] > 0).bool()] = -100

    # ---- HF logits ----
    with torch.inference_mode():
        hf_out = hf(
            input_ids=input_ids,
            pixel_values=pixel_values_hf,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            use_cache=False,
        )
    hf_logits = hf_out.logits[0].float()

    # ---- our logits (training forward) ----
    pv_ours = (
        pixel_values_hf.reshape(1, n_patches, patch_size, patch_size, 3)
        .permute(0, 1, 4, 2, 3)
        .reshape(1, n_patches, 3 * patch_size * patch_size)
        .contiguous()
        .unsqueeze(0)
    )
    pooled_idx_ours = image_token_pooling.unsqueeze(0)
    with torch.inference_mode():
        our_logits_full = ours(
            input_ids=input_ids, images=pv_ours, pooled_patches_idx=pooled_idx_ours
        )
    our_logits = our_logits_full[0].float()[:, : hf_logits.shape[-1]]

    # (a) logit parity on response positions.
    resp = (loss_masks[0] > 0).bool()
    diff = (hf_logits[resp] - our_logits[resp]).abs().max().item()
    assert diff < 5.0, f"{model_id}: response logit max abs diff = {diff:.3e}"

    # (b) float-weighted loss parity (mm_olmo reduction): independently reduce HF logits
    # and compare to our weighted_cross_entropy_loss on our logits.
    flat_labels = labels[0]
    per_token_hf = torch.nn.functional.cross_entropy(
        hf_logits, flat_labels, ignore_index=-100, reduction="none"
    )
    ref_ce = torch.dot(per_token_hf, loss_masks[0])
    our_ce, _ = weighted_cross_entropy_loss(
        our_logits, flat_labels, loss_masks[0], ignore_index=-100
    )
    rel = (our_ce - ref_ce).abs().item() / max(ref_ce.abs().item(), 1.0)
    assert rel < 0.05, f"{model_id}: weighted CE rel diff = {rel:.3e} (our={our_ce}, ref={ref_ce})"


# ---------------------------------------------------------------------------
# 2. Subsegment branch isolation (CPU, tiny model)
# ---------------------------------------------------------------------------


def _packed_two_branch_inputs(seed: int = 0):
    """A tiny text-only packed 2-branch example (no images) + the prefix length."""
    rng = np.random.RandomState(seed)
    prefix = [2, 3, 4, 5]  # arbitrary non-image text tokens (last is carry-over)
    branch0 = rng.randint(6, _LM_VOCAB, size=4).tolist()
    branch1 = rng.randint(6, _LM_VOCAB, size=3).tolist()
    seq = build_packed_sequence(prefix, [branch0, branch1], eos_id=1)
    return seq, len(prefix)


def test_subsegment_branch_isolation():
    """Changing branch 0's tokens must not change branch 1's logits, but must
    change branch 0's own logits (sibling isolation via the subsegment ``and_mask``)."""
    torch.manual_seed(0)
    model = _tiny_multimodal_cfg().build(init_device="cpu").eval()

    seq, n_prefix = _packed_two_branch_inputs()
    input_ids = torch.tensor(seq["input_ids"]).unsqueeze(0)
    subsegment_ids = torch.tensor(seq["subsegment_ids"]).unsqueeze(0)
    position_ids = torch.tensor(seq["position_ids"]).unsqueeze(0)
    assert ATTEND_ALL_SUBSEGMENT_ID in subsegment_ids.tolist()[0]

    branch1_mask = subsegment_ids[0] == 1
    branch0_mask = subsegment_ids[0] == 0

    with torch.inference_mode():
        logits_a = model(
            input_ids=input_ids, subsegment_ids=subsegment_ids, position_ids=position_ids
        )

    # Perturb branch-0 tokens only.
    input_ids_b = input_ids.clone()
    perturb_pos = torch.nonzero(branch0_mask)[-1].item()  # a response token in branch 0
    input_ids_b[0, perturb_pos] = (input_ids_b[0, perturb_pos] + 7) % _LM_VOCAB
    with torch.inference_mode():
        logits_b = model(
            input_ids=input_ids_b, subsegment_ids=subsegment_ids, position_ids=position_ids
        )

    # Branch 1 logits unchanged (cannot attend branch 0).
    b1_diff = (logits_a[0][branch1_mask] - logits_b[0][branch1_mask]).abs().max().item()
    assert b1_diff < 1e-4, f"branch-1 logits changed by {b1_diff:.3e} when branch-0 perturbed"

    # Branch 0 logits at/after the perturbed position DO change (sanity).
    b0_diff = (logits_a[0][branch0_mask] - logits_b[0][branch0_mask]).abs().max().item()
    assert b0_diff > 1e-4, "branch-0 logits did not change when its own tokens were perturbed"


def test_subsegment_branches_match_independent_sequences():
    """A packed branch must produce the same logits as forwarding that branch as a
    standalone (prefix + branch) sequence — confirming positions + masking are right."""
    torch.manual_seed(0)
    model = _tiny_multimodal_cfg().build(init_device="cpu").eval()

    prefix = [2, 3, 4, 5]
    branch0 = [10, 11, 12, 13]
    branch1 = [20, 21, 22]
    packed = build_packed_sequence(prefix, [branch0, branch1], eos_id=1)
    input_ids = torch.tensor(packed["input_ids"]).unsqueeze(0)
    subseg = torch.tensor(packed["subsegment_ids"]).unsqueeze(0)
    pos = torch.tensor(packed["position_ids"]).unsqueeze(0)
    with torch.inference_mode():
        packed_logits = model(input_ids=input_ids, subsegment_ids=subseg, position_ids=pos)[0]

    # Standalone branch 0: prefix + branch0 (carry-over already included as prefix[-1]).
    standalone0 = torch.tensor(prefix + branch0).unsqueeze(0)
    with torch.inference_mode():
        logits0 = model(input_ids=standalone0)[0]

    # In the packed sequence, branch 0 occupies positions [n_prefix-1 .. ] starting with
    # the carry-over token. Compare the branch-0 response logits.
    branch0_idx = torch.nonzero(torch.tensor(packed["subsegment_ids"]) == 0).flatten()
    # Standalone branch-0 response positions are the last len(branch0) positions.
    standalone_resp = logits0[-len(branch0) :]
    packed_resp = packed_logits[branch0_idx][-len(branch0) :]
    diff = (standalone_resp - packed_resp).abs().max().item()
    assert diff < 1e-4, f"packed vs standalone branch-0 logits diff = {diff:.3e}"


def test_root_subsegments_weighting_end_to_end():
    """The packed example's float loss weights match ``1/sqrt(n_branches)`` on response
    tokens and feed a finite weighted loss through the tiny model."""
    torch.manual_seed(0)
    model = _tiny_multimodal_cfg().build(init_device="cpu").eval()

    prefix = [2, 3, 4, 5]
    packed = build_packed_sequence(prefix, [[10, 11], [20, 21, 22]], eos_id=1)
    w = packed["loss_masks"]
    nonzero = w[w > 0]
    assert np.allclose(nonzero, 1.0 / np.sqrt(2)), nonzero

    input_ids = torch.tensor(packed["input_ids"]).unsqueeze(0)
    subseg = torch.tensor(packed["subsegment_ids"]).unsqueeze(0)
    pos = torch.tensor(packed["position_ids"]).unsqueeze(0)
    labels = torch.tensor(packed["labels"]).reshape(-1)
    weights = torch.tensor(packed["loss_masks"]).reshape(-1)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, subsegment_ids=subseg, position_ids=pos)[0]
    ce, _ = weighted_cross_entropy_loss(logits, labels, weights, ignore_index=-100)
    assert torch.isfinite(ce)


def test_image_splice_forward_with_subsegments():
    """End-to-end smoke: an image-bearing single-branch example forwards and yields a
    finite weighted loss through the tiny model (image splice + loss path)."""
    torch.manual_seed(0)
    model = _tiny_multimodal_cfg().build(init_device="cpu").eval()

    # 1 pooled image token (4 patches), then text; loss on the text tail.
    input_ids, images, pooled_idx = _make_inputs(batch=1, seq_len=8)
    token_type_ids = (input_ids == _IMAGE_PATCH_TOKEN).long()
    labels = torch.full_like(input_ids, -100)
    labels[0, :-1] = input_ids[0, 1:]
    loss_masks = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_masks[0, 4:] = 1.0  # loss on text tokens after the image token
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            images=images,
            pooled_patches_idx=pooled_idx,
            token_type_ids=token_type_ids,
        )[0]
    ce, _ = weighted_cross_entropy_loss(
        logits, labels.reshape(-1), loss_masks.reshape(-1), ignore_index=-100
    )
    assert torch.isfinite(ce)


def test_response_logits_only_matches_full_logits():
    """Flattened response-only logits match the full forward on loss positions."""
    torch.manual_seed(0)
    model = _tiny_multimodal_cfg().build(init_device="cpu").eval()

    prefix = [2, 3, 4, 5]
    packed = build_packed_sequence(prefix, [[10, 11, 12], [20, 21]], eos_id=1)
    input_ids = torch.tensor(packed["input_ids"]).unsqueeze(0)
    subseg = torch.tensor(packed["subsegment_ids"]).unsqueeze(0)
    pos = torch.tensor(packed["position_ids"]).unsqueeze(0)
    loss_masks = torch.tensor(packed["loss_masks"], dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        full_logits = model(input_ids=input_ids, subsegment_ids=subseg, position_ids=pos)[0]
        narrow_logits = model(
            input_ids=input_ids,
            subsegment_ids=subseg,
            position_ids=pos,
            response_logits_only=True,
            loss_masks=loss_masks,
        )

    resp = loss_masks[0] > 0
    torch.testing.assert_close(narrow_logits, full_logits[resp], atol=1e-5, rtol=1e-5)


def test_response_logits_only_variable_response_counts_per_row():
    """Two rows with different response lengths: narrowed logits match full forward."""
    torch.manual_seed(1)
    model = _tiny_multimodal_cfg().build(init_device="cpu").eval()

    B, S = 2, 16
    input_ids = torch.randint(6, _LM_VOCAB, (B, S))
    loss_masks = torch.zeros(B, S, dtype=torch.float32)
    loss_masks[0, [2, 5, 8, 12]] = 1.0
    loss_masks[1, [1, 7]] = 1.0

    with torch.inference_mode():
        full_logits = model(input_ids=input_ids)
        narrow_logits = model(
            input_ids=input_ids,
            response_logits_only=True,
            loss_masks=loss_masks,
        )

    resp = loss_masks > 0
    torch.testing.assert_close(
        narrow_logits,
        full_logits.view(-1, full_logits.shape[-1])[resp.view(-1)],
        atol=1e-5,
        rtol=1e-5,
    )
    assert int(resp.sum()) == 6
