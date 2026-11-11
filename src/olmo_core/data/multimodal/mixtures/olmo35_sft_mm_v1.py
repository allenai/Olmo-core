"""Stage-2 image data for Olmo 3.5 multimodal SFT, version 1.

Molmo2's stage-2 SFT mixture (:mod:`.image_only_v9`) without its text group (the text SFT data
replaces it) and adjusted for what the Olmo 3.5 mid-training checkpoints already trained on in
stage 1 (the v6 visual mixture):

* **dropped**: sources stage 1 trained on as they are here, DVQA, FigureQA, PlotQA and the seven
  single-document CoSyn sets;
* **anchors**: PixMo-Cap and PixMo-Clocks, also stage-1 sources, kept at a small fixed dose so
  the chat-format versions of those tasks are not lost;
* **kept at Molmo2's rates**: every other source. The pointing and counting sources use the same
  images as stage 1 but stage-2's own annotations and prompts; the rest are new to stage 1.

Every source renders in the Olmo 3.5 chat template of the text SFT recipe (``message_format=
"olmo35"``, :mod:`~olmo_core.data.multimodal.olmo35_layout`), image rows thinking off.

The dose is set in image-data tokens (default :data:`DEFAULT_DOSE_TOKENS`). Given each source's
measured mean tokens per example (:data:`MEAN_TOKENS`), :func:`expected_examples` gives the
examples each source contributes at that dose; they are the ``visual_example_weights`` of the
mixed mid-training recipe, and :data:`MEAN_LOSS_WEIGHTS` its calibrated ``mean_loss_weight``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from olmo_core.config import Config
from olmo_core.data.multimodal.olmo35_layout import OLMO35_TOKEN_IDS
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .image_only_v9 import IMAGE_ONLY_V9_SUBMIXTURES, get_image_only_v9_source

__all__ = [
    "ANCHOR_TOKENS",
    "DEFAULT_DOSE_TOKENS",
    "DROPPED",
    "KEPT",
    "MEAN_LOSS_WEIGHTS",
    "MEAN_TOKENS",
    "SEQUENCE_LENGTH",
    "SOURCE_CLASS",
    "V9_EXAMPLE_RATES",
    "build_source_config",
    "build_sources",
    "expected_examples",
    "visual_loss_mass",
]

SEQUENCE_LENGTH = 65_536
"""The SFT sequence length (the text SFT recipe's)."""

DEFAULT_DOSE_TOKENS = 3_500_000_000
"""Image-data tokens consumed over the run, by default."""

DROPPED: Tuple[str, ...] = (
    "dv_qa",
    "figure_qa",
    "plot_qa",
    "cosyn_chart_exp",
    "cosyn_chemical_exp",
    "cosyn_diagram_exp",
    "cosyn_document",
    "cosyn_math_exp",
    "cosyn_music_exp",
    "cosyn_table_exp",
)
"""Stage-2 sources the stage-1 v6 mixture trained on with the same images and questions."""

ANCHOR_TOKENS: Dict[str, int] = {"pixmo_cap": 85_000_000, "pixmo_clocks": 40_000_000}
"""Fixed token doses of the two stage-1 sources kept as anchors."""

SOURCE_CLASS: Dict[str, str] = {
    **{name: "dropped" for name in DROPPED},
    **{name: "anchor" for name in ANCHOR_TOKENS},
    **{
        name: "stage-1 images, same task"
        for name in (
            "pixmo_points_train",
            "pixmo_count_train",
            "pixmo_points_high_freq_train",
            "cosyn_point",
        )
    },
    **{
        name: "stage-1 images, new task"
        for name in ("pixmo_cap_qa_as_user_qa", "pixmo_multi_points", "text_vqa")
    },
}
"""Each stage-2 image source's relation to stage 1 (``new`` when absent), from the image overlap
inventory (``src/scripts/olmo35_sft_mm_overlap.py``, receipt
``/weka/oe-training-default/jasonr/mmsft-data/overlap-v1/overlap.json``): the pointing and counting
sources share 100% of their images with stage-1's PixMo-Points v2, PixMo-Count v2 and CoSyn-Point
v2; CapQA 99.6% with PixMo-Cap and multi-image points 98% with PixMo-Points v2 (other tasks on the
same images); TextVQA 99% with TextOCR (questions instead of transcription). AMA (0.2%),
correction QA, ST-VQA (0%) and the sources with no stage-1 pool in common are new."""

KEPT: Tuple[str, ...] = tuple(
    source.name
    for group in IMAGE_ONLY_V9_SUBMIXTURES
    if group.name != "nlp"
    for source in group.datasets
    if source.name not in DROPPED
)
"""The v1 sources, in image_only_v9's order."""

V9_EXAMPLE_RATES: Dict[str, float] = {
    "pixmo_ask_model_anything": 0.05433875786,
    "pixmo_cap": 0.06442384949,
    "pixmo_cap_qa_as_user_qa": 0.08780405836,
    "correction_qa_multi_only_max5": 0.04343333428,
    "coco_2014_vqa_multi": 0.01999002186,
    "text_vqa": 0.01292389109,
    "okvqa": 0.006594486477,
    "chart_qa_weighted": 0.01168767787,
    "doc_qa": 0.0138018638,
    "info_qa": 0.01075125537,
    "ai2_diagram_v2_mix_transparent": 0.008521096442,
    "a_okvqa_mc": 0.00907363452,
    "a_okvqa_da": 0.00907363452,
    "science_qa_img": 0.0054785805,
    "tabwmp_da": 0.01055025419,
    "st_qa": 0.01099629934,
    "tally_qa": 0.02484402322,
    "mantis_instruct_llava_665k_multi_multi_only": 0.027385236,
    "mantis_instruct_nlvr2_multi_only": 0.02041886937,
    "mantis_instruct_spot-the-diff_multi_only": 0.006014901325,
    "pixmo_clocks": 0.03473863047,
    "cosyn_multidoc_chart_exp": 0.01263034692,
    "cosyn_multidoc_chemical_exp": 0.003392312199,
    "cosyn_multidoc_diagram_exp": 0.008437697984,
    "cosyn_multidoc_doc_exp": 0.01154890981,
    "cosyn_multidoc_music_exp": 0.004785868891,
    "cosyn_multidoc_table_exp": 0.008251722762,
    "pixmo_multi_points": 0.04366498051,
    "pixmo_points_train": 0.03926487031,
    "pixmo_count_train": 0.02653020399,
    "pixmo_points_high_freq_train": 0.03106955907,
    "cosyn_point": 0.02547038611,
}
"""image_only_v9's flat example rates (mm_olmo SubMixture math over the dataset lengths, text
group included in the normalization), from ``src/scripts/olmo35_sft_mm_probe.py``."""

MEAN_TOKENS: Dict[str, float] = {
    "pixmo_ask_model_anything": 1249.4062,
    "pixmo_cap": 1232.8438,
    "pixmo_cap_qa_as_user_qa": 1268.5469,
    "correction_qa_multi_only_max5": 2801.6875,
    "coco_2014_vqa_multi": 888.0547,
    "text_vqa": 1132.7109,
    "okvqa": 789.6484,
    "chart_qa_weighted": 1050.1172,
    "doc_qa": 1079.7344,
    "info_qa": 1208.5,
    "ai2_diagram_v2_mix_transparent": 836.6172,
    "a_okvqa_mc": 814.8672,
    "a_okvqa_da": 792.4922,
    "science_qa_img": 761.8438,
    "tabwmp_da": 507.1406,
    "st_qa": 645.7891,
    "tally_qa": 771.7578,
    "mantis_instruct_llava_665k_multi_multi_only": 3600.5078,
    "mantis_instruct_nlvr2_multi_only": 1722.0547,
    "mantis_instruct_spot-the-diff_multi_only": 1023.7969,
    "pixmo_clocks": 1029.9062,
    "cosyn_multidoc_chart_exp": 4483.4453,
    "cosyn_multidoc_chemical_exp": 3150.9453,
    "cosyn_multidoc_diagram_exp": 4515.6719,
    "cosyn_multidoc_doc_exp": 4499.0703,
    "cosyn_multidoc_music_exp": 3286.6094,
    "cosyn_multidoc_table_exp": 4981.2812,
    "pixmo_multi_points": 3790.4531,
    "pixmo_points_train": 1762.2266,
    "pixmo_count_train": 1561.5078,
    "pixmo_points_high_freq_train": 1431.2422,
    "cosyn_point": 1705.3516,
}
"""Mean tokens per example in the ``olmo35`` layout at :data:`SEQUENCE_LENGTH` (128-row probe,
seed 0, the tokenizer of the text SFT recipe), measured 2026-11-11 by
``src/scripts/olmo35_sft_mm_probe.py``; receipt
``/weka/oe-training-default/jasonr/mmsft-data/probe-v1/probe.json``."""

MEAN_LOSS_WEIGHTS: Dict[str, float] = {
    "pixmo_ask_model_anything": 222.828125,
    "pixmo_cap": 216.09375,
    "pixmo_cap_qa_as_user_qa": 203.109375,
    "correction_qa_multi_only_max5": 220.398438,
    "coco_2014_vqa_multi": 17.25,
    "text_vqa": 4.554688,
    "okvqa": 3.945312,
    "chart_qa_weighted": 3.178227,
    "doc_qa": 6.632812,
    "info_qa": 4.78125,
    "ai2_diagram_v2_mix_transparent": 5.476562,
    "a_okvqa_mc": 4.929688,
    "a_okvqa_da": 3.8125,
    "science_qa_img": 7.492188,
    "tabwmp_da": 3.71875,
    "st_qa": 4.914062,
    "tally_qa": 5.53125,
    "mantis_instruct_llava_665k_multi_multi_only": 153.570312,
    "mantis_instruct_nlvr2_multi_only": 4.265625,
    "mantis_instruct_spot-the-diff_multi_only": 17.757812,
    "pixmo_clocks": 11.664062,
    "cosyn_multidoc_chart_exp": 618.203125,
    "cosyn_multidoc_chemical_exp": 580.101562,
    "cosyn_multidoc_diagram_exp": 620.523438,
    "cosyn_multidoc_doc_exp": 637.617188,
    "cosyn_multidoc_music_exp": 537.007812,
    "cosyn_multidoc_table_exp": 651.578125,
    "pixmo_multi_points": 26.032816,
    "pixmo_points_train": 68.435942,
    "pixmo_count_train": 7.179687,
    "pixmo_points_high_freq_train": 49.004694,
    "cosyn_point": 21.420313,
}
"""Mean summed ``loss_masks`` per example (the mixture's ``mean_loss_weight``), same probe. Below
the mean loss tokens where a source's ``message_weight`` is 0.2 (the pointing sources)."""


def build_source_config(
    name: str,
    *,
    seed: int = 0,
    token_ids: Molmo2TokenIds = OLMO35_TOKEN_IDS,
    max_sequence_length: int = SEQUENCE_LENGTH,
    annotation_sampling: str = "all",
) -> Config:
    """The config of one stage-2 source in the ``olmo35`` layout.

    Mirrors :func:`~.image_only_v9.build_image_only_v9_dataset`'s per-source settings (styles,
    message weights, high-resolution rates), with the Olmo 3.5 chat layout and image token IDs.

    :param name: An image_only_v9 source name.
    :param seed: Dataset seed.
    :param token_ids: Image token IDs (the mixture overwrites them with the model's).
    :param max_sequence_length: The mixture's sequence length; also the token budget of an
        example whose annotations become documents.
    :param annotation_sampling: For the single-image pointing sources: ``"all"`` (every
        annotation, as later turns of one conversation about the image), ``"documents"`` (each its
        own document sharing the crops, the image's tokens repeated per annotation) or ``"one"``
        (one annotation drawn per example and epoch).
    """
    from olmo_core.data.multimodal.academic_dataset import AcademicDatasetConfig
    from olmo_core.data.multimodal.multi_image_datasets import (
        CorrectionQaDatasetConfig,
        CoSynMultiDocDatasetConfig,
        MantisInstructDatasetConfig,
        PixMoMultiPointsDatasetConfig,
    )
    from olmo_core.data.multimodal.paths import PIXMO_DATASETS
    from olmo_core.data.multimodal.pixmo_ama import PixMoAmaDatasetConfig
    from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
    from olmo_core.data.multimodal.pixmo_cap_qa import PixMoCapQaDatasetConfig
    from olmo_core.data.multimodal.pixmo_points import (
        CoSynPointDatasetConfig,
        PixMoCountDatasetConfig,
        PixMoPointsDatasetConfig,
    )

    if name in DROPPED or name == "tulu4":
        raise KeyError(f"{name} is not a v1 source")
    src = get_image_only_v9_source(name)
    if src is None:
        raise KeyError(f"Unknown stage-2 source {name}")
    common = dict(seed=seed, token_ids=token_ids, message_format="olmo35")
    if name == "pixmo_ask_model_anything":
        return PixMoAmaDatasetConfig(**common)
    if name == "pixmo_cap":
        return PixMoCapDatasetConfig(
            dataset_path=f"{PIXMO_DATASETS}/cap",
            mode="sft_demo",
            max_sequence_length=max_sequence_length,
            **common,
        )
    if name == "pixmo_cap_qa_as_user_qa":
        return PixMoCapQaDatasetConfig(**common)
    if name == "correction_qa_multi_only_max5":
        return CorrectionQaDatasetConfig(**common)
    if name.startswith("mantis_instruct_"):
        subset = name[len("mantis_instruct_") :].replace("_multi_only", "")
        return MantisInstructDatasetConfig(subset=subset, **common)
    if name.startswith("cosyn_multidoc_"):
        doc_type = name[len("cosyn_multidoc_") :].replace("_exp", "")
        return CoSynMultiDocDatasetConfig(
            doc_type=doc_type, use_exp=name.endswith("_exp"), **common
        )
    pointing = dict(
        message_weight=src.message_weight,
        p_high_res=src.override_p_high_res or 0.0,
        loss_token_weighting="none",
        **common,
    )
    if name == "pixmo_multi_points":
        return PixMoMultiPointsDatasetConfig(**pointing)
    single = dict(
        max_sequence_length=max_sequence_length,
        annotation_sampling=annotation_sampling,
        **pointing,
    )
    if name == "pixmo_points_train":
        return PixMoPointsDatasetConfig(kind="basic", **single)
    if name == "pixmo_points_high_freq_train":
        return PixMoPointsDatasetConfig(kind="high_frequency", **single)
    if name == "pixmo_count_train":
        # One annotation per image: nothing to sample.
        return PixMoCountDatasetConfig(max_sequence_length=max_sequence_length, **pointing)
    if name == "cosyn_point":
        return CoSynPointDatasetConfig(**single)
    return AcademicDatasetConfig(name=name, message_weight=src.message_weight, **common)


def build_sources(**kwargs) -> Dict[str, Config]:
    """Every v1 source's config, keyed by its image_only_v9 name (``kwargs`` as
    :func:`build_source_config`)."""
    return {name: build_source_config(name, **kwargs) for name in KEPT}


@dataclass(frozen=True)
class _Dose:
    examples: Dict[str, float]
    tokens: Dict[str, float]


def _dose(
    dose_tokens: float,
    rates: Mapping[str, float],
    mean_tokens: Mapping[str, float],
    anchor_tokens: Mapping[str, float],
) -> _Dose:
    anchors = {name: anchor_tokens[name] / mean_tokens[name] for name in anchor_tokens}
    rest = {name: rates[name] for name in KEPT if name not in anchor_tokens}
    budget = dose_tokens - sum(anchor_tokens.values())
    if budget <= 0:
        raise ValueError("the dose must exceed the anchors' tokens")
    per_unit = sum(rate * mean_tokens[name] for name, rate in rest.items())
    scale = budget / per_unit
    examples = {**{name: rate * scale for name, rate in rest.items()}, **anchors}
    tokens = {name: n * mean_tokens[name] for name, n in examples.items()}
    return _Dose(examples={name: examples[name] for name in KEPT}, tokens=tokens)


def expected_examples(
    dose_tokens: float = DEFAULT_DOSE_TOKENS,
    *,
    rates: Optional[Mapping[str, float]] = None,
    mean_tokens: Optional[Mapping[str, float]] = None,
    anchor_tokens: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Expected examples each v1 source contributes at ``dose_tokens`` image-data tokens.

    The anchors take their fixed token doses; the other sources share the rest in proportion to
    image_only_v9's example rates. Pass the result as the mixture's ``visual_example_weights``.
    """
    return _dose(
        dose_tokens,
        V9_EXAMPLE_RATES if rates is None else rates,
        MEAN_TOKENS if mean_tokens is None else mean_tokens,
        ANCHOR_TOKENS if anchor_tokens is None else anchor_tokens,
    ).examples


def visual_loss_mass(
    dose_tokens: float = DEFAULT_DOSE_TOKENS,
    *,
    mean_loss_weights: Optional[Mapping[str, float]] = None,
    **kwargs,
) -> float:
    """Expected summed loss weight of the image data at ``dose_tokens`` (in loss tokens, weighted).

    With the text source's expected summed loss weight ``T`` over the run, the mixture's text
    loss share is ``T / (T + visual_loss_mass(dose))``.
    """
    means = MEAN_LOSS_WEIGHTS if mean_loss_weights is None else mean_loss_weights
    return sum(n * means[name] for name, n in expected_examples(dose_tokens, **kwargs).items())
