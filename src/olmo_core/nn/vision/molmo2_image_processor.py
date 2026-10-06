"""
Molmo2 image preprocessor — ported from mm_olmo with no mm_olmo/einops/torchvision dependencies.

Drop-in replacement for ``Molmo2ImageProcessor`` from
``olmo.hf_model.image_processing_molmo2``.  All logic is pure numpy + torch.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image as PILImage

# Tolerate truncated/corrupt images (PIL otherwise raises ``OSError: image file is truncated``
# from ``.load()``, which kills the data-worker thread and — under distributed packing — hangs
# the other ranks' collective into a NCCL watchdog abort). The PixMo image sets contain a few
# truncated files; mm_olmo sets the same flag. This is a process-global PIL setting, so importing
# this module (done by every dataset's image path) enables it for all in-process worker threads.
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:  # PIL is an optional dep; image datasets won't load without it anyway
    pass

# SigLIP / IMAGENET_STANDARD mean and std (same as transformers.image_utils.IMAGENET_STANDARD_*)
_IMAGE_MEAN = [0.5, 0.5, 0.5]
_IMAGE_STD = [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# Internal helpers (private)
# ---------------------------------------------------------------------------


def _normalize_image(arr: np.ndarray, mean: list, std: list) -> np.ndarray:
    """In-place normalize (h, w, 3) float32 array."""
    arr -= np.array(mean, dtype=np.float32)[None, None, :]
    arr /= np.array(std, dtype=np.float32)[None, None, :]
    return arr


def _resize_image(arr: np.ndarray, target_hw: list) -> np.ndarray:
    """
    Resize ``arr`` (h, w, 3) to ``target_hw = [new_h, new_w]``.

    Accepts float32 [0, 1] or uint8 [0, 255]; always returns float32 [0, 1].
    Uses ``torch.nn.functional.interpolate`` (bilinear, antialias=False) to
    match the behaviour of ``torchvision.transforms.Resize`` with the same
    flags used in mm_olmo.
    """
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, h, w)
    if arr.dtype == np.uint8:
        t = t / 255.0
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False, antialias=False)
    t = torch.clamp(t, 0.0, 1.0)
    return t.squeeze(0).permute(1, 2, 0).numpy()  # (h, w, 3) float32 [0, 1]


def _select_tiling(h: int, w: int, patch_size: int, max_crops: int) -> np.ndarray:
    """Return ``(th, tw)`` tiling that best fits an image of size ``(h, w)``."""
    tilings = []
    for i in range(1, max_crops + 1):
        for j in range(1, max_crops + 1):
            if i * j <= max_crops:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([h, w], dtype=np.float32)
    with np.errstate(divide="ignore"):
        required_scale_d = (candidate_resolutions.astype(np.float32) / original_size,)
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)
    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def _arange_for_pooling(idx_arr: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    """
    Group patch indices for pooling.

    Equivalent to::

        einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)

    with symmetric -1 padding to make dimensions divisible.

    :param idx_arr: ``(H, W)`` int32 array of patch indices (-1 = padding/mask).
    :returns: ``(ph, pw, pool_h * pool_w)`` grouped indices.
    """
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    H, W = idx_arr.shape
    ph = H // pool_h
    pw = W // pool_w
    return (
        idx_arr.reshape(ph, pool_h, pw, pool_w)
        .transpose(0, 2, 1, 3)
        .reshape(ph, pw, pool_h * pool_w)
    )


def _batch_pixels_to_patches_cf(array: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Reshape image crops to patch sequences in **channel-first** order.

    :param array: ``(n_crops, h, w, 3)`` float32.
    :param patch_size: spatial patch size (e.g. 14).
    :returns: ``(n_crops, n_patches, c * patch_size * patch_size)`` where the
              patch dimension is ``(c, kh, kw)`` — channel-first.
    """
    n_crops, h, w, c = array.shape
    hp = h // patch_size
    wp = w // patch_size
    array = array.reshape(n_crops, hp, patch_size, wp, patch_size, c)
    # → (n_crops, hp, wp, c, patch_size, patch_size)  [channel-first per patch]
    array = array.transpose(0, 1, 3, 5, 2, 4)
    return array.reshape(n_crops, hp * wp, c * patch_size * patch_size)


def _build_resized_image(
    image: np.ndarray,
    base_hw: list,
    mean: list,
    std: list,
    patch_size: int,
) -> tuple:
    """
    Build the global (low-res) crop.

    :returns: ``(resized, resize_idx)`` where *resized* is ``(1, h, w, 3)``
              float32 and *resize_idx* is ``(h_patches, w_patches)`` int32.
    """
    resized = _resize_image(image, base_hw)
    resized = _normalize_image(resized, mean, std)
    resized = resized[np.newaxis]  # (1, h, w, 3)
    crop_patch_w = base_hw[1] // patch_size
    crop_patch_h = base_hw[0] // patch_size
    resize_idx = np.arange(crop_patch_w * crop_patch_h, dtype=np.int32).reshape(
        crop_patch_h, crop_patch_w
    )
    return resized, resize_idx


def _build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    margins: list,
    base_hw: list,
    mean: list,
    std: list,
    patch_size: int,
) -> tuple:
    """
    Decompose ``image`` into overlapping high-res tiles.

    :returns: ``(crop_arr, patch_idx_arr)`` where *crop_arr* is
              ``(n_crops, h, w, 3)`` and *patch_idx_arr* is
              ``(resized_h_patches, resized_w_patches)`` int32 (−1 = overlap).
    """
    crop_size = base_hw[0]
    assert base_hw[0] == base_hw[1]
    left_margin, right_margin = margins

    total_margin_px = patch_size * (left_margin + right_margin)
    crop_window_patches = crop_size // patch_size - (left_margin + right_margin)
    crop_window_size = crop_window_patches * patch_size
    crop_patch_h = base_hw[0] // patch_size
    crop_patch_w = base_hw[1] // patch_size
    orig_h, orig_w = image.shape[:2]

    tiling = _select_tiling(
        orig_h - total_margin_px,
        orig_w - total_margin_px,
        crop_window_size,
        max_crops,
    )

    src = _resize_image(
        image,
        [
            tiling[0] * crop_window_size + total_margin_px,
            tiling[1] * crop_window_size + total_margin_px,
        ],
    )
    src = _normalize_image(src, mean, std)

    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)

    on_crop = 0
    for i in range(tiling[0]):
        y0 = i * crop_window_size
        for j in range(tiling[1]):
            x0 = j * crop_window_size
            crop_arr[on_crop] = src[y0 : y0 + crop_size, x0 : x0 + crop_size]
            patch_idx = np.arange(crop_patch_w * crop_patch_h, dtype=np.int32).reshape(
                crop_patch_h, crop_patch_w
            )
            patch_idx += on_crop * crop_patch_h * crop_patch_w
            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0] - 1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1] - 1:
                patch_idx[:, -right_margin:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # Reorder patch_idx_arr to left-to-right reading order
    patch_idx_arr = patch_idx_arr.reshape(tiling[0], tiling[1], crop_patch_h, crop_patch_w)
    patch_idx_arr = patch_idx_arr.transpose(0, 2, 1, 3)
    patch_idx_arr = patch_idx_arr.reshape(-1)
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
        src.shape[0] // patch_size,
        src.shape[1] // patch_size,
    )
    return crop_arr, patch_idx_arr


def _image_to_patches_and_grids(
    image: np.ndarray,
    max_crops: int,
    margins: list,
    base_hw: list,
    mean: list,
    std: list,
    patch_size: int,
    pool_h: int,
    pool_w: int,
) -> tuple:
    """
    Full preprocessing pipeline for a single image.

    :returns: ``(image_grid, crops_cf, pooling_idx)`` where

              * *image_grid* — ``(1, 4)`` int32: ``[resized_h, resized_w, h, w]``
              * *crops_cf*   — ``(n_crops, n_patches, patch_dim)`` channel-first float32
              * *pooling_idx* — ``(n_pool_tokens, pool_h*pool_w)`` int32
    """
    crop_patch_h = base_hw[0] // patch_size
    crop_patch_w = base_hw[1] // patch_size

    crop_arr, patch_idx_arr = _build_overlapping_crops(
        image, max_crops, margins, base_hw, mean, std, patch_size
    )

    pooling_idx = _arange_for_pooling(patch_idx_arr, pool_h, pool_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape(-1, pool_h * pool_w)

    resized, resize_idx = _build_resized_image(image, base_hw, mean, std, patch_size)
    all_crops = np.concatenate([resized, crop_arr], 0)  # global crop first

    resize_pooling = _arange_for_pooling(resize_idx, pool_h, pool_w)
    resized_h, resized_w = resize_pooling.shape[:2]
    resize_pooling = resize_pooling.reshape(-1, pool_h * pool_w)

    # Offset high-res patch indices past the global crop's patches
    pooling_idx = np.where(pooling_idx >= 0, pooling_idx + crop_patch_h * crop_patch_w, -1)
    pooling_idx = np.concatenate([resize_pooling, pooling_idx], 0)

    image_grid = np.stack([np.array([resized_h, resized_w, h, w], dtype=np.int32)], 0)  # (1, 4)

    return image_grid, _batch_pixels_to_patches_cf(all_crops, patch_size), pooling_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_image_molmo2(
    pil_img: "PILImage.Image",
    dtype: torch.dtype,
    device: torch.device,
    *,
    image_size: int = 378,
    patch_size: int = 14,
    max_crops: int = 8,
    high_res_max_crops: int = 24,
    p_high_res: float = 0.0,
    is_training: bool = True,
    rng: np.random.RandomState | None = None,
    overlap_margins: tuple = (4, 4),
    pool_h: int = 2,
    pool_w: int = 2,
    mean: list = _IMAGE_MEAN,
    std: list = _IMAGE_STD,
) -> tuple:
    """
    Preprocess a PIL image for Molmo2 :class:`~olmo_core.nn.vision.MultimodalLM`.

    This replaces ``Molmo2ImageProcessor`` from ``mm_olmo`` with no external
    dependencies beyond numpy, torch, and PIL.

    :param pil_img: Input PIL image (any mode; converted to RGB internally).
    :param dtype: Floating-point dtype for the returned image tensor.
    :param device: Target device.
    :param image_size: Square base crop size in pixels (default 378).
    :param patch_size: ViT patch size in pixels (default 14).
    :param max_crops: Maximum number of high-res crops (default 8).
    :param high_res_max_crops: Crop budget when high-res mode is sampled (default 24).
    :param p_high_res: Training-time probability of using ``high_res_max_crops`` (default 0).
    :param is_training: When False, always uses ``max_crops``.
    :param rng: Optional ``numpy.random.RandomState`` for high-res sampling.
    :param overlap_margins: ``(left, right)`` overlap margins in patches (default ``(4, 4)``).
    :param pool_h: Pooling height (default 2).
    :param pool_w: Pooling width (default 2).
    :param mean: Per-channel normalisation mean (default SigLIP ``[0.5, 0.5, 0.5]``).
    :param std: Per-channel normalisation std (default SigLIP ``[0.5, 0.5, 0.5]``).

    :returns images: ``(1, n_crops, n_patches, patch_dim)`` channel-first float tensor.
    :returns pooling_idx: ``(1, n_pool_tokens, pool_h * pool_w)`` int64 tensor (−1 = pad).
    :returns image_grid: ``np.ndarray`` of shape ``(4,)`` with
                         ``[resized_h, resized_w, h, w]`` pooled-grid dimensions.
    """
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32) / 255.0  # (h, w, 3) float32 [0, 1]

    base_hw = [image_size, image_size]
    margins = list(overlap_margins)
    crop_budget = max_crops
    if is_training and p_high_res > 0:
        rng = rng if rng is not None else np.random.RandomState()
        if rng.random() < p_high_res:
            crop_budget = max(max_crops, high_res_max_crops)

    image_grid_batch, crops_cf, pooling_idx = _image_to_patches_and_grids(
        arr,
        max_crops=crop_budget,
        margins=margins,
        base_hw=base_hw,
        mean=mean,
        std=std,
        patch_size=patch_size,
        pool_h=pool_h,
        pool_w=pool_w,
    )

    # Add batch dimension
    images_t = torch.from_numpy(crops_cf[np.newaxis]).to(dtype=dtype, device=device)
    pooling_t = torch.from_numpy(pooling_idx[np.newaxis].astype(np.int64)).to(device=device)

    return images_t, pooling_t, image_grid_batch[0]
