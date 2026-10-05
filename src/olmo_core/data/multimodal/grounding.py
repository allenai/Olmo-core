"""Point/grounding formatting for Molmo2 pointing & counting data.

Dependency-free port of the single-image path of ``mm_olmo``'s
``GroundingPreprocessor`` (``olmo/models/molmo2/grounding_formatter.py``, the
``html-v2`` format used by stage-1) plus the pointing / counting answer assembly
(``get_point_string`` in ``olmo/data/data_formatter.py``).

A set of points for one image is rendered as::

    <points coords="1 1 XXX YYY 2 XXX YYY ...">label</points>

where the leading ``1`` is the (single) image index, then ``PTID X Y`` per point,
with X/Y scaled to 0-1000 (3-digit, zero-padded) and points sorted by (x, y).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "format_points_tag",
    "format_multi_image_points_tag",
    "pointing_answer",
    "multi_image_pointing_answer",
    "POINTING_PROMPTS",
    "POINT_COUNT_PROMPTS",
    "MULTI_IMAGE_POINTING_PROMPTS",
    "MULTI_IMAGE_POINT_COUNT_PROMPTS",
]


def _scale_point(x: float, y: float) -> Tuple[int, int]:
    """Clamp to [0,1] then scale to 0-1000 integers (matches ``_scale_point``)."""
    x = min(max(float(x), 0.0), 1.0)
    y = min(max(float(y), 0.0), 1.0)
    return round(1000 * x), round(1000 * y)


def format_points_tag(points_norm: Sequence[Sequence[float]], label: str) -> str:
    """Render normalized (0-1) points for a single image as an html-v2 ``<points>`` tag.

    :param points_norm: ``(N, 2)`` array-like of already-normalized ``(x, y)`` in [0, 1].
    :param label: The object label / text inside the tag.

    :returns: ``<points coords="1 1 XXX YYY ...">label</points>``; empty string if no points.
    """
    pts = [_scale_point(x, y) for x, y in points_norm]
    if not pts:
        return ""
    # Sort by (x, y) after rounding (matches build_single_image_coordinates).
    pts.sort()
    body = " ".join(f"{i} {x:03d} {y:03d}" for i, (x, y) in enumerate(pts, start=1))
    coord_str = f"1 {body}"  # leading "1" is the single image index
    return f'<points coords="{coord_str}">{label}</points>'


def pointing_answer(
    points_norm: Sequence[Sequence[float]],
    label: str,
    style: str,
    count: Optional[int] = None,
) -> str:
    """Assemble the assistant answer text (matches ``get_point_string``).

    :param points_norm: normalized ``(N, 2)`` points.
    :param label: object label.
    :param style: ``"pointing"``/``"point"``/``"cosyn_point"``/``"gui_point"`` (just the
        points tag) or ``"point_count"`` (``Counting the <points…> shows a total of N.``).
    :param count: number of points; defaults to ``len(points_norm)``.
    """
    n = len(points_norm) if count is None else count
    if n == 0:
        return "There are none."
    tag = format_points_tag(points_norm, label)
    if style in ("point_count", "point_then_count"):
        return f"Counting the {tag} shows a total of {n}."
    if style in ("count_then_point", "count_point"):
        return f"There are {n} {tag}."
    if style == "count":
        return str(n)
    # "pointing" / "point" / "cosyn_point" / "gui_point" / None
    return tag


def normalize_points(
    xy: np.ndarray, point_scale: Optional[float], image_size: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Normalize raw ``(N, 2)`` points to [0, 1] (matches ``normalize_coordinates``).

    :param xy: raw ``(N, 2)`` points.
    :param point_scale: if set, divide by this scalar (e.g. 100 for 0-100 percent coords).
    :param image_size: ``(w, h)`` to divide by when ``point_scale`` is None (pixel coords).
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return xy.reshape(0, 2)
    if point_scale is not None:
        return xy / float(point_scale)
    assert image_size is not None, "image_size required when point_scale is None (pixel coords)"
    w, h = image_size
    out = xy.copy()
    out[:, 0] /= float(w)
    out[:, 1] /= float(h)
    return out


# Prompt pools (verbatim from mm_olmo GENERAL_PROMPTS_V1; one is sampled per branch,
# `{label}` filled with the object label). data_formatter.py:257 / :302.
POINTING_PROMPTS: Tuple[str, ...] = (
    "Point to {label}\nPlease say 'There are none.' if it is not in the image.",
    'Point to all occurrences of "{label}"',
    "Point to any {label} in the image",
    "Point to any {label} in the image.",
    "Point: Where are the {label}",
    "Show me where the {label} are",
    "Can you show me where the {label} are?",
    "Show me where the {label} are",
    "Show me where a {label} is",
    "Show me where a {label} is.",
    "If there are any {label} in the image? Show me where they are.",
    "Where are the {label}?",
    "Generate a list of points showing where the {label} are.",
    'Find the "{label}".',
    'Find a "{label}".',
    "Locate all {label}.",
    "Locate an {label}.",
    "Locate a {label}.",
    "Locate every {label}.",
    "Locate {label}.",
    "Locate the {label}.",
    "Object: {label}\nInstruction: Point to the object.",
    "find {label}",
    "find {label}.",
    "Point to every {label}",
    "find any {label} in the picture",
    "Find the {label}",
    "Find any {label}",
    "Point to a {label}",
    "Point to an {label}",
    "Look for {label} in the image and show me where they are.",
    "Help me find an object in the image by pointing to them.\nObject: {label}.",
    "I am looking for {label}, where can they be found in the image?",
    "Can you see any {label} in the image? Point to them.",
    "Point out each {label} in the image.",
    "Point out every {label} in the image.",
    "Point to the {label} in the image.",
    "Locate each {label} in the image.",
    "Can you point out all {label} in this image?",
    "Please find {label} and show me where they are.",
    "If there are any {label} present, indicate their positions.",
    "If there is a {label} present, indicate its positions.",
    "show me all visible {label}",
)
POINT_COUNT_PROMPTS: Tuple[str, ...] = (
    "How many {label} are there?",
    "How many {label}?",
    "How many {label}.",
    "how many {label}.",
    "how many {label}?",
    'How many "{label}" are there in the image?',
    "How many {label} are there in the image?",
    "Tell me how many {label} there are",
    "Tell me how many {label} there are and point to them.",
    "how many {label}",
    "Tell me where each {label} is.",
    "Tell me how many {label} are in the image",
    "count {label}",
    "count every {label}",
    "count each {label}",
    "count {label}.",
    "Count the {label}.",
    "How many {label} do you see?",
    "How many {label} are visible?",
    "Count all the {label}",
    "how mmny {label}?",
    "Count every {label} in the picture.",
    "Count all the {label}",
    "Count each {label}",
    "Point to and count the {label} in the picture.",
    "Point and count {label}",
    "Point to every {label}",
    "Locate the {label} and count them",
    "Locate every {label} and count them",
    "Find all the {label}. How many are there?",
    "Find each {label}. How many are there?",
    "Point at {label} and then tell me the count.",
    "What is the total number of {label} in the image?",
    "What is the number of {label}?",
    "In this image, how many {label} are there?",
    "In all the picture, how many {label} are there?",
    "Point at the {label} and then count them.",
    "Point to all the visible {label} output the total count.",
    "Point to all the {label} visible and output the total count. \nPlease say 'There are none.' if it is not in the image.",
    'Point to all occurrences of "{label}" and output the total count.',
    "Show me where the {label} are and output the total count.",
    "Where are the {label}? How many are there?",
    "Generate list of points showing where the {label} are and output the total count.",
    "Object: {label}\nInstruction: Point to the object and output the total count.",
    "find any {label} in the picture and output the total count.",
    "Can you see any {label} in the image? Point to them and output the total count.",
    "Can you point out all {label} in this image? How many are there?",
    "If there are any {label} present, indicate their positions and output the total count.",
    "How many {label} are there in the image? Point to them and output the total count.",
    "How many {label} are there in the image?",
    "Give me the count of {label} in the image.",
    "How many {label} are visible in the image?",
    "How many {label} are there?",
    "In the image, how many {label} are there?",
    "Can you count the number of {label} in the image?",
    "Can you count every {label} in the picture?",
    "Can you see any {label} in the image? How many are there?",
    "Are there any {label} in the image? How many are there?",
    "If you see any {label} in the image, give me the count. Otherwise, say 'There are none.'",
    "Object: {label}\nInstruction: How many are there?",
)


# ---------------------------------------------------------------------------
# Multi-image pointing (mm_olmo UnifiedPointFormatter.format_multi_image_points,
# pointing_format="html-v2": image groups joined by ";")
# ---------------------------------------------------------------------------

# Per-image group separator inside ``coords="..."``. mm_olmo html-v1 uses "\t",
# html-v2 (Molmo2 stage-2, ``get_model``) uses ";". Single-image answers are
# identical under both.
MULTI_IMAGE_COORD_SEP = ";"


def format_multi_image_points_tag(
    points_by_image: Sequence[Tuple[int, Sequence[Sequence[float]]]],
    label: str,
) -> str:
    """Render normalized points spanning several images as one ``<points>`` tag.

    Matches mm_olmo ``_build_multi_image_coordinates``: each image contributes
    ``"{image_index} {i x y ...}"`` with per-image (x, y)-sorted points and point
    indices *continuing* across images; groups are joined by
    :data:`MULTI_IMAGE_COORD_SEP`.

    :param points_by_image: ``[(image_index_1based, points_norm), ...]`` in display
        order; ``points_norm`` are already-normalized ``(x, y)`` in [0, 1].
    :param label: text inside the tag.
    """
    on = 1
    groups = []
    for image_index, pts_norm in points_by_image:
        pts = [_scale_point(x, y) for x, y in pts_norm]
        pts.sort()
        body = " ".join(f"{i} {x:03d} {y:03d}" for i, (x, y) in enumerate(pts, start=on))
        on += len(pts)
        groups.append(f"{image_index} {body}")
    coord_str = MULTI_IMAGE_COORD_SEP.join(groups)
    return f'<points coords="{coord_str}">{label}</points>'


def multi_image_pointing_answer(
    points_by_image: Sequence[Tuple[int, Sequence[Sequence[float]]]],
    label: str,
    style: str,
) -> str:
    """Assemble the assistant answer for multi-image pointing styles.

    Same phrasing as the single-image :func:`pointing_answer`
    (mm_olmo ``build_point_output``), with the count summed over all images.
    """
    n = sum(len(p) for _, p in points_by_image)
    if n == 0:
        return "There are none."
    tag = format_multi_image_points_tag(points_by_image, label)
    if style in ("multi_image_point_then_count", "point_then_count", "point_count"):
        return f"Counting the {tag} shows a total of {n}."
    if style in ("multi_image_count_then_point", "count_then_point"):
        return f"There are {n} {tag}."
    if style in ("multi_image_counting", "count"):
        return str(n)
    return tag


# Verbatim active entries of mm_olmo GENERAL_PROMPTS_V1["multi_image_pointing"] /
# ["multi_image_point_then_count"] (data_formatter.py:414-460; commented-out
# entries omitted). `{selected_images}` is "all images" or "image_1, image_2, ...".
MULTI_IMAGE_POINTING_PROMPTS: Tuple[str, ...] = (
    "Find {selected_label} in {selected_images}.",
    "find {selected_label} in {selected_images}.",
    "Point to {selected_label} in {selected_images}.",
    "Point to any {selected_label} in {selected_images}.",
    "Point to all {selected_label} in {selected_images}.",
    'Point to all occurrences of "{selected_label}" in {selected_images}.',
    "Can you point to {selected_label} in {selected_images}?",
    "Show me where the {selected_label} are in {selected_images}?",
    "Show me where a {selected_label} is in {selected_images}?",
    "Show me where a {selected_label} is in {selected_images}.",
    "In {selected_images}, point to {selected_label}.",
)
MULTI_IMAGE_POINT_COUNT_PROMPTS: Tuple[str, ...] = (
    "How many {selected_label} are there in {selected_images}?",
    "how many {selected_label} are there in {selected_images}?",
    "How many {selected_label} are there in {selected_images}.",
    "count {selected_label} in {selected_images}?",
    "count every {selected_label} in {selected_images}?",
    "count each {selected_label} in {selected_images}?",
    "count {selected_label} in {selected_images}.",
    "Count the {selected_label} in {selected_images}.",
    "Point then count {selected_label} in {selected_images}.",
    "Point and count {selected_label} in {selected_images}.",
    "Can you point and count {selected_label} in {selected_images}?",
    "Can you point then count {selected_label} in {selected_images}?",
    "Point to {selected_label} in {selected_images}, then count them.",
    "Point to all {selected_label} in {selected_images}, then count them.",
)
