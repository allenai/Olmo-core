"""Default visual sources for bridge, perception, and joint alignment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from olmo_core.config import Config
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

DEFAULT_ALIGNMENT_ARTIFACT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts"
)

ALIGNMENT_LOSS_TARGETS: dict[str, dict[str, float]] = {
    "bridge": {"pixmo_caption": 0.70, "pixmo_transcript": 0.30},
    "perception": {
        "pixmo_caption": 0.45,
        "pixmo_transcript": 0.20,
        "pixmo_points_basic": 0.10,
        "pixmo_points_high_frequency": 0.02,
        "cosyn_point": 0.03,
        "ocr_document": 0.10,
        "scalar_count": 0.05,
        "audited_alignment": 0.05,
    },
    "joint": {
        "native_text_replay": 0.35,
        "pixmo_caption": 0.28,
        "pixmo_transcript": 0.12,
        "pixmo_points_basic": 0.05,
        "pixmo_points_high_frequency": 0.01,
        "cosyn_point": 0.02,
        "ocr_document": 0.08,
        "count_numeric": 0.04,
        "audited_alignment": 0.05,
    },
}

ALIGNMENT_MEAN_LOSS_WEIGHTS: dict[str, dict[str, float]] = {
    "bridge": {
        "pixmo_caption": 28.829104933305643,
        "pixmo_transcript": 29.77422616124386,
    },
    "perception": {
        "audited_alignment": 18.860581716464367,
        "cosyn_point": 19.9274699697271,
        "ocr_document": 4.121176112443209,
        "pixmo_caption": 28.593906218127813,
        "pixmo_points_basic": 33.50284379025106,
        "pixmo_points_high_frequency": 27.439346029539593,
        "pixmo_transcript": 29.704384807148017,
        "scalar_count": 3.464101552963257,
    },
    "joint": {
        "audited_alignment": 18.266647445358103,
        "cosyn_point": 20.201689065172104,
        "count_numeric": 3.464101552963257,
        "native_text_replay": 8191.0,
        "ocr_document": 4.189212302095257,
        "pixmo_caption": 29.160138869367074,
        "pixmo_points_basic": 33.28980797799886,
        "pixmo_points_high_frequency": 28.820370947258198,
        "pixmo_transcript": 29.65188992714684,
    },
}
"""Loss-weight calibration at 8,192 tokens for each phase."""

ALIGNMENT_ONE_ANNOTATION_MEAN_LOSS_WEIGHTS: dict[str, dict[str, float]] = {
    "perception": {
        "cosyn_point": 9.25648039940279,
        "pixmo_points_basic": 9.418736778199673,
        "pixmo_points_high_frequency": 18.28113580151694,
    },
    "joint": {
        "cosyn_point": 9.25648039940279,
        "pixmo_points_basic": 9.418736778199673,
        "pixmo_points_high_frequency": 18.28113580151694,
    },
}
"""Calibration at 8,192 tokens of the multi-annotation sources when each example keeps one
sampled annotation (``annotation_sampling="one"``, used for language models with document
boundaries). The other sources keep :data:`ALIGNMENT_MEAN_LOSS_WEIGHTS`."""

ALIGNMENT_ARTIFACT_MANIFESTS: dict[str, str] = {
    "pixmo-cap-content-disjoint-v1/build-state.json": (
        "31a03bc22d2a2bfb04ac1d4a1d0b0626879cf05e269f2e1bd63679102f7acb72"
    ),
    "pixmo-cap-content-disjoint-v1/vision-alignment-validation-manifest.json": (
        "83cb9594648952c53d3ad042605a2e099e6a1e04bb24fed18625c1db53452d42"
    ),
    "perception-provenance-v2/build-state.json": (
        "23e2970fed5805f20a9acf13cb088a22723d1469dc52810fc4be1c9b500442fa"
    ),
    "perception-provenance-v2/vision-alignment-perception-provenance.json": (
        "73cb3920676db5e16d789f7257800dcb44b2553b6463cff81beb740213d921e2"
    ),
    "finevision-materialization-v1/build-plan.json": (
        "c074b71c1c234cb92d0f3d8b2c83b6dadb2eef13a672b92dc8cca33158d74ea0"
    ),
    "finevision-materialization-v1/vision-alignment-finevision-materialization.json": (
        "1436ad9d3f67d4e66a4f6e8e5f02c16a074af4707062ea28af6bacf89aada063"
    ),
}
"""SHA-256 of the build manifests of the prepared artifacts the calibration was measured on,
relative to the artifact root."""


def has_calibrated_artifacts(artifact_root: str, phase: str) -> bool:
    """Whether ``artifact_root`` holds the prepared artifacts the calibration of ``phase`` was
    measured on: the default root, or a copy whose build manifests are byte-identical.

    :param artifact_root: Directory containing the prepared alignment datasets and selections.
    :param phase: ``bridge``, ``perception``, or ``joint``.
    """
    if artifact_root == DEFAULT_ALIGNMENT_ARTIFACT_ROOT:
        return True
    for name, digest in ALIGNMENT_ARTIFACT_MANIFESTS.items():
        if phase == "bridge" and not name.startswith("pixmo-cap-content-disjoint-v1/"):
            continue  # bridge reads only the caption artifacts
        path = Path(artifact_root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True


def build_visual_sources(
    phase: str,
    sequence_length: int,
    artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    *,
    split: str = "train",
) -> dict[str, Config]:
    """Configure visual sources and their prepared image-disjoint selections.

    Source configs support recipe and CLI overrides. This function reads selection metadata
    without loading datasets or images.

    :param phase: ``bridge``, ``perception``, or ``joint``.
    :param sequence_length: Maximum serialized example length.
    :param artifact_root: Directory containing the prepared alignment datasets and selections.
    :param split: Logical ``train`` or ``validation`` split.
    :returns: Dataset configs keyed by source name.
    """
    if phase not in ALIGNMENT_LOSS_TARGETS:
        raise ValueError(f"Unknown alignment phase {phase!r}")
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown alignment split {split!r}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    root = Path(artifact_root)
    common: dict[str, Any] = {
        "max_crops": 8,
        "max_sequence_length": sequence_length,
        "loss_token_weighting": "root_subsegments_root_tokens",
        "message_format": "document",
        "seed": 0,
    }

    sources: dict[str, Config] = {
        "pixmo_caption": PixMoCapDatasetConfig(
            dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
            split=split,
            require_split=True,
            mode="caption",
            fixed_prompt="Description:",
            style_length_conditioning=False,
            **common,
        ),
        "pixmo_transcript": PixMoCapDatasetConfig(
            dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
            split=split,
            require_split=True,
            mode="transcript",
            require_transcript=True,
            fixed_prompt="Transcript:",
            style_length_conditioning=False,
            **common,
        ),
    }
    if phase == "bridge":
        return sources
    extend = _PHASE_SOURCE_EXTENSIONS.get(phase)
    if extend is None:
        raise ValueError(
            f"Visual sources for the {phase!r} phase are provided by the alignment phases layer "
            "(olmo_core.internal.vision_alignment_phases); import it to register them."
        )
    return extend(sources, root, common, phase, split)


_PHASE_SOURCE_EXTENSIONS: dict[str, Callable[..., dict[str, Config]]] = {}
"""Perception and joint source builders, registered by the phases layer so the bridge path never
imports their data adapters."""


def register_phase_sources(phase: str, builder: Callable[..., dict[str, Config]]) -> None:
    """Register the visual-source builder of a non-bridge phase."""
    _PHASE_SOURCE_EXTENSIONS[phase] = builder


# ---------------------------------------------------------------------------------------------
# Stage-1 v3 data (``--recipe.data=stage1_v3``)
# ---------------------------------------------------------------------------------------------

STAGE1_V3_GROUPS: dict[str, tuple[str, ...]] = {
    "caption": ("pixmo_caption", "pixmo_transcript"),
    "pointing": ("pixmo_points_v2", "pixmo_count_v2", "cosyn_point_v2"),
    "ocr": (
        "olmocr_documents",
        "olmocr_books",
        "olmocr_loc_transcripts",
        "olmocr_national_archives",
        "text_rich_chart",
        "text_rich_diagram",
        "text_rich_doc",
        "text_rich_graphic",
        "text_rich_table",
        "textocr",
        "nvidia_synth_en",
        "synth_receipts_en",
    ),
    "academic": (
        "cosyn_chart_exp",
        "cosyn_chemical_exp",
        "cosyn_diagram_exp",
        "cosyn_document",
        "cosyn_math_exp",
        "cosyn_music_exp",
        "cosyn_table_exp",
        "dv_qa",
        "figure_qa",
        "plot_qa",
    ),
    "clocks": ("pixmo_clocks",),
}
"""The Molmo2-Stage1 ``v3`` groups, as perception/joint sources (Molmo2-Stage1's ``pixmo_cap``
is split into its caption and transcript branches)."""

STAGE1_V3_SOURCES: tuple[str, ...] = tuple(n for g in STAGE1_V3_GROUPS.values() for n in g)

_STAGE1_V3_SEED = 95818
"""Molmo2-Stage1's ``data_seed``: the seed of its caption, academic and clock sources."""


def build_stage1_v3_sources(
    phase: str,
    sequence_length: int,
    artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
) -> dict[str, Config]:
    """Configure the Molmo2-Stage1 ``v3`` mixture's sources for a perception or joint phase.

    Each source is configured as ``src/scripts/train/Molmo2-Stage1.py --recipe=v3`` configures
    it (prompt tags, audit styles, negative sub-sampling, OCR and academic source lists), except:

    * examples are serialized as plain documents (``message_format="document"``) and truncated
      at ``sequence_length``; the mixture sets the image token IDs of the language model;
    * every source weights response tokens equally (``loss_token_weighting="none"``) with no
      per-source message weight: the per-source loss shares are set by the mixture's targets
      (:data:`STAGE1_V3_LOSS_TARGETS`) instead;
    * PixMo-Cap's caption and transcript are two sources (``long_caption:`` and
      ``transcript:``) over the prepared alignment caption selection, so the image-disjoint
      validation split stays disjoint.

    Multi-annotation sources (pointing, counting, figure captions, academic QA) keep
    ``annotation_sampling="all"`` here; the recipe switches them to one sampled annotation per
    example for language models with document boundaries.

    :param phase: ``perception`` or ``joint``.
    :param sequence_length: Maximum serialized example length.
    :param artifact_root: Directory with the prepared alignment caption dataset and selections.
    :returns: Dataset configs keyed by source name, in :data:`STAGE1_V3_SOURCES` order.
    """
    import json

    from olmo_core.data.multimodal.academic_dataset import Stage1AcademicDatasetConfig
    from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
    from olmo_core.data.multimodal.mixtures.ocr import (
        OCR_TAR_SOURCES,
        OLMOCR_MIX_SOURCES,
        TEXT_RICH_SOURCES,
    )
    from olmo_core.data.multimodal.ocr_caption_tars import OcrCaptionTarsDatasetConfig
    from olmo_core.data.multimodal.olmocr import OlmOcrMixDatasetConfig
    from olmo_core.data.multimodal.paths import OE_ENCODER_DATA
    from olmo_core.data.multimodal.pixmo_points import (
        COSYN_POINT_V2_PATH,
        CoSynPointDatasetConfig,
    )
    from olmo_core.data.multimodal.pixmo_points_v2 import (
        PixMoCountV2DatasetConfig,
        PixMoPointsV2DatasetConfig,
    )
    from olmo_core.data.multimodal.synthetic_ocr import (
        NvidiaSynthOcrDatasetConfig,
        SyntheticReceiptsDatasetConfig,
    )
    from olmo_core.data.multimodal.text_rich_caption import TextRichCaptionDatasetConfig

    if phase not in ("perception", "joint"):
        raise ValueError(f"Stage-1 v3 data is for the perception and joint phases, not {phase!r}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    root = Path(artifact_root)
    common: dict[str, Any] = {
        "max_crops": 8,
        "max_sequence_length": sequence_length,
        "loss_token_weighting": "none",
        "message_format": "document",
    }
    # The caption sources read the prepared caption selection, as the alignment sources do.
    selection_root = root / "perception-provenance-v2"
    with (selection_root / "vision-alignment-perception-provenance.json").open() as stream:
        manifest = json.load(stream)

    def caption_source(name: str, **kwargs: Any) -> Config:
        selection = manifest["sources"][name]["train"]
        return MultimodalSourceConfig(
            dataset=PixMoCapDatasetConfig(
                dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
                split=selection["physical_split"],
                require_split=True,
                style_tag=True,
                seed=_STAGE1_V3_SEED,
                **kwargs,
                **common,
            ),
            selection_path=str(selection_root / selection["selection"]["path"]),
        )

    # Molmo2-Stage1's POINTING_V2_* settings: audit-failed point sets kept behind `aux_*` tags,
    # two easy negatives and a quarter of the paired negatives per image and epoch.
    pointing_audit_style = ("aux_point_count", "aux_pointing")
    sources: dict[str, Config] = {
        "pixmo_caption": caption_source("pixmo_caption", mode="caption"),
        "pixmo_transcript": caption_source(
            "pixmo_transcript", mode="transcript", require_transcript=True
        ),
        "pixmo_points_v2": PixMoPointsV2DatasetConfig(
            p_paired_negatives=0.25,
            n_easy_samples=2,
            audit_style=pointing_audit_style,
            filter_audit=False,
            **common,
        ),
        "pixmo_count_v2": PixMoCountV2DatasetConfig(
            audit_style=pointing_audit_style, filter_audit=False, **common
        ),
        "cosyn_point_v2": CoSynPointDatasetConfig(
            dataset_path=COSYN_POINT_V2_PATH,
            audit_style="aux_cosyn_point",
            prompt_templates="none",
            system_prompt="style_and_length_v2",
            **common,
        ),
    }
    for name, subset in OLMOCR_MIX_SOURCES.items():
        sources[name] = OlmOcrMixDatasetConfig(subset=subset, **common)
    for name, category in TEXT_RICH_SOURCES.items():
        sources[name] = TextRichCaptionDatasetConfig(category=category, **common)
    textocr = OCR_TAR_SOURCES["textocr"]
    sources["textocr"] = OcrCaptionTarsDatasetConfig(
        dataset_path=str(Path(OE_ENCODER_DATA) / textocr.relpath),
        style=textocr.style,
        strip_text_tags=textocr.strip_text_tags,
        **common,
    )
    sources["nvidia_synth_en"] = NvidiaSynthOcrDatasetConfig(**common)
    sources["synth_receipts_en"] = SyntheticReceiptsDatasetConfig(**common)
    for name in STAGE1_V3_GROUPS["academic"] + STAGE1_V3_GROUPS["clocks"]:
        sources[name] = Stage1AcademicDatasetConfig(name=name, seed=_STAGE1_V3_SEED, **common)
    if tuple(sources) != STAGE1_V3_SOURCES:
        raise RuntimeError(f"Stage-1 v3 sources {list(sources)} != {list(STAGE1_V3_SOURCES)}")
    return sources


def build_stage1_v3_validation_sources(
    sequence_length: int, artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT
) -> dict[str, Config]:
    """Caption and transcript validation in the stage-1 v3 prompt form (``long_caption:`` /
    ``transcript:``), over the prepared image-disjoint caption validation selection.

    :param sequence_length: Maximum serialized example length.
    :param artifact_root: Directory with the prepared alignment caption dataset and selections.
    """
    import json

    from olmo_core.data.multimodal.alignment import MultimodalSourceConfig

    root = Path(artifact_root)
    selection_root = root / "perception-provenance-v2"
    with (selection_root / "vision-alignment-perception-provenance.json").open() as stream:
        manifest = json.load(stream)
    out: dict[str, Config] = {}
    for name, source, kwargs in (
        ("v3_long_caption", "pixmo_caption", {"mode": "caption"}),
        ("v3_transcript", "pixmo_transcript", {"mode": "transcript", "require_transcript": True}),
    ):
        selection = manifest["sources"][source]["validation"]
        out[name] = MultimodalSourceConfig(
            dataset=PixMoCapDatasetConfig(
                dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
                split=selection["physical_split"],
                require_split=True,
                style_tag=True,
                max_crops=8,
                max_sequence_length=sequence_length,
                loss_token_weighting="none",
                message_format="document",
                seed=0,
                **kwargs,
            ),
            selection_path=str(selection_root / selection["selection"]["path"]),
        )
    return out


STAGE1_V3_EXAMPLE_RATES: dict[str, float] = {
    "pixmo_caption": 0.32999999999999996,
    "pixmo_transcript": 0.32999999999999996,
    "pixmo_points_v2": 0.10836976774217352,
    "pixmo_count_v2": 0.018153129054673138,
    "cosyn_point_v2": 0.03347710320315334,
    "olmocr_documents": 0.051913124219767806,
    "olmocr_books": 0.013652104646113742,
    "olmocr_loc_transcripts": 0.01093646785071678,
    "olmocr_national_archives": 0.010991507167651411,
    "text_rich_chart": 0.040571169388423985,
    "text_rich_diagram": 0.025764369994072833,
    "text_rich_doc": 0.04517827828681451,
    "text_rich_graphic": 0.020153404131816606,
    "text_rich_table": 0.03833277819887207,
    "textocr": 0.016365994104406242,
    "nvidia_synth_en": 0.051913124219767806,
    "synth_receipts_en": 0.014227677791576233,
    "cosyn_chart_exp": 0.02636166459695638,
    "cosyn_chemical_exp": 0.007293612988017891,
    "cosyn_diagram_exp": 0.014422140678059707,
    "cosyn_document": 0.02059279616047255,
    "cosyn_math_exp": 0.01992204297925949,
    "cosyn_music_exp": 0.008438287456894922,
    "cosyn_table_exp": 0.016635501567530055,
    "dv_qa": 0.00771303642602405,
    "figure_qa": 0.00771303642602405,
    "plot_qa": 0.010907880720760916,
    "pixmo_clocks": 0.03,
}
"""Per-source example-sampling rates of Molmo2-Stage1 ``--recipe=v3`` (the mixture weights its
``_build_mixture_sources`` returns for the v3 run's source sizes; logged in that run as "Mixture
sources / sizes / weights"). ``pixmo_caption`` and ``pixmo_transcript`` both carry the rate of its
one ``pixmo_cap`` source, whose examples hold a caption and a transcript branch each."""

STAGE1_V3_REFERENCE_MEAN_LOSS_WEIGHTS: dict[str, float] = {
    "pixmo_caption": 274.384765625,
    "pixmo_transcript": 286.85546875,
    "pixmo_points_v2": 502.0625,
    "pixmo_count_v2": 47.28125,
    "cosyn_point_v2": 126.328125,
    "olmocr_documents": 688.1328125,
    "olmocr_books": 434.1015625,
    "olmocr_loc_transcripts": 254.71875,
    "olmocr_national_archives": 287.265625,
    "text_rich_chart": 655.2890625,
    "text_rich_diagram": 685.3984375,
    "text_rich_doc": 863.21875,
    "text_rich_graphic": 524.359375,
    "text_rich_table": 822.5390625,
    "textocr": 53.9453125,
    "nvidia_synth_en": 167.765625,
    "synth_receipts_en": 207.6171875,
    "cosyn_chart_exp": 482.4375,
    "cosyn_chemical_exp": 286.2890625,
    "cosyn_diagram_exp": 341.3203125,
    "cosyn_document": 52.484375,
    "cosyn_math_exp": 410.25,
    "cosyn_music_exp": 257.0859375,
    "cosyn_table_exp": 500.9140625,
    "dv_qa": 26.0,
    "figure_qa": 27.046875,
    "plot_qa": 167.3984375,
    "pixmo_clocks": 12.765625,
}
"""Mean loss weight per example of each source on the v3 Stage-1 path: the sources built exactly as
``Molmo2-Stage1.py --recipe=v3`` builds them (Qwen3 tokenizer, every annotation as a sibling
branch, ``loss_token_weighting="none"``, caption ``message_weight`` 1.25, 2,560-token
truncation), ``sum(loss_masks)`` over 128 rows per source drawn with
``numpy.random.default_rng(0)`` (as :meth:`MultimodalMixtureConfig.estimate_mean_loss_weights`
draws), source epoch 0. ``pixmo_cap``'s mass is split by branch: ``pixmo_caption`` is the caption
branch's mean, ``pixmo_transcript`` the transcript branch's (274.4 + 286.9 = 561.2)."""

_V3_LOSS_MASS = {
    name: STAGE1_V3_EXAMPLE_RATES[name] * STAGE1_V3_REFERENCE_MEAN_LOSS_WEIGHTS[name]
    for name in STAGE1_V3_SOURCES
}
STAGE1_V3_LOSS_TARGETS: dict[str, float] = {
    name: mass / sum(_V3_LOSS_MASS.values()) for name, mass in _V3_LOSS_MASS.items()
}
"""Target loss mass of each stage-1 v3 source: its expected share of the v3 Stage-1 run's loss
(rate x mean loss weight per example, normalized). By group: caption 39.2%, pointing 12.6%, OCR
39.3%, academic QA 8.8%, clocks 0.08% (the Molmo2-Stage1 recipe notes, from an 80-example sample:
39.2 / 12.8 / 38.9 / 9.1 / 0.08)."""

STAGE1_V3_MEAN_LOSS_WEIGHTS: dict[str, float] = {
    "pixmo_caption": 208.78125,
    "pixmo_transcript": 225.8046875,
    "pixmo_points_v2": 28.5625,
    "pixmo_count_v2": 34.2265625,
    "cosyn_point_v2": 21.9375,
    "olmocr_documents": 766.2734375,
    "olmocr_books": 506.1484375,
    "olmocr_loc_transcripts": 345.171875,
    "olmocr_national_archives": 288.4453125,
    "text_rich_chart": 204.3828125,
    "text_rich_diagram": 223.1015625,
    "text_rich_doc": 276.5703125,
    "text_rich_graphic": 158.296875,
    "text_rich_table": 262.3515625,
    "textocr": 50.953125,
    "nvidia_synth_en": 183.015625,
    "synth_receipts_en": 171.2421875,
    "cosyn_chart_exp": 47.109375,
    "cosyn_chemical_exp": 45.6015625,
    "cosyn_diagram_exp": 39.6953125,
    "cosyn_document": 5.515625,
    "cosyn_math_exp": 405.0703125,
    "cosyn_music_exp": 37.3203125,
    "cosyn_table_exp": 50.9375,
    "dv_qa": 2.3125,
    "figure_qa": 3.0,
    "plot_qa": 6.0,
    "pixmo_clocks": 10.7421875,
}
"""Calibration of the stage-1 v3 sources as the alignment recipe builds them for a document-mode
LM (:func:`build_stage1_v3_sources` with ``annotation_sampling="one"``, dolma2 tokenizer at the
pinned revision, 8,192 tokens): :meth:`MultimodalMixtureConfig.estimate_mean_loss_weights` with
128 samples per source, seed 0. Perception and joint build the same sources."""
