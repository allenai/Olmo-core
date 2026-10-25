"""image-only-v9 mixture registry for Molmo2 SFT dataset parity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from olmo_core.data.multimodal.academic_dataset import AcademicDatasetConfig
from olmo_core.data.multimodal.mixture_weights import (
    DatasetSource,
    SubMixture,
    compute_flat_mixture_weights,
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
from olmo_core.data.multimodal.tulu import Tulu4DatasetConfig

__all__ = [
    "IMAGE_ONLY_V9_SUBMIXTURES",
    "DEBUG_MIXTURE_DATASETS",
    "VALIDATION_MIXTURES",
    "build_image_only_v9_datasets",
    "build_image_only_v9_dataset",
    "build_image_only_v9_mixture",
]

# Parity-validated subset for stage-2 smoke tests before the full 32-source sweep is green.
DEBUG_MIXTURE_DATASETS: Tuple[str, ...] = ("tulu4", "text_vqa", "chart_qa_weighted")

DEMO_MIXTURE_DATASETS: Tuple[str, ...] = (
    "pixmo_ask_model_anything",
    "pixmo_cap",
    "pixmo_cap_qa_as_user_qa",
    "correction_qa_multi_only_max5",
)
POINTING_MIXTURE_DATASETS: Tuple[str, ...] = (
    "pixmo_multi_points",
    "pixmo_points_train",
    "pixmo_count_train",
    "pixmo_points_high_freq_train",
    "cosyn_point",
)
NLP_MIXTURE_DATASETS: Tuple[str, ...] = ("tulu4",)
MULTI_IMAGE_MIXTURE_DATASETS: Tuple[str, ...] = (
    "correction_qa_multi_only_max5",
    "mantis_instruct_llava_665k_multi_multi_only",
    "mantis_instruct_nlvr2_multi_only",
    "mantis_instruct_spot-the-diff_multi_only",
    "cosyn_multidoc_chart_exp",
    "cosyn_multidoc_chemical_exp",
    "cosyn_multidoc_diagram_exp",
    "cosyn_multidoc_doc_exp",
    "cosyn_multidoc_music_exp",
    "cosyn_multidoc_table_exp",
    "pixmo_multi_points",
)

IMAGE_ONLY_V9_SUBMIXTURES: List[SubMixture] = [
    SubMixture(
        "demo",
        0.25,
        [
            DatasetSource("pixmo_ask_model_anything"),
            DatasetSource("pixmo_cap", root_size_factor=100_000),
            DatasetSource("pixmo_cap_qa_as_user_qa"),
            DatasetSource("correction_qa_multi_only_max5"),
        ],
    ),
    SubMixture(
        "image_academic",
        0.418,
        [
            DatasetSource("coco_2014_vqa_multi"),
            DatasetSource("text_vqa"),
            DatasetSource("okvqa"),
            DatasetSource("chart_qa_weighted"),
            DatasetSource("doc_qa"),
            DatasetSource("info_qa"),
            DatasetSource("ai2_diagram_v2_mix_transparent"),
            DatasetSource("a_okvqa_mc"),
            DatasetSource("a_okvqa_da"),
            DatasetSource("science_qa_img"),
            DatasetSource("tabwmp_da"),
            DatasetSource("st_qa"),
            DatasetSource("tally_qa"),
            # Multi-image (mm_olmo IMAGE_ACADEMIC_V2)
            DatasetSource("mantis_instruct_llava_665k_multi_multi_only"),
            DatasetSource("mantis_instruct_nlvr2_multi_only"),
            DatasetSource("mantis_instruct_spot-the-diff_multi_only"),
            DatasetSource("pixmo_clocks", root_size_factor=250_000),
            DatasetSource("dv_qa", root_size_factor=10_000),
            DatasetSource("figure_qa", root_size_factor=10_000),
            DatasetSource("plot_qa", root_size_factor=20_000),
            DatasetSource("cosyn_chart_exp"),
            DatasetSource("cosyn_chemical_exp"),
            DatasetSource("cosyn_diagram_exp"),
            DatasetSource("cosyn_document"),
            DatasetSource("cosyn_math_exp"),
            DatasetSource("cosyn_music_exp"),
            DatasetSource("cosyn_table_exp"),
            DatasetSource("cosyn_multidoc_chart_exp"),
            DatasetSource("cosyn_multidoc_chemical_exp"),
            DatasetSource("cosyn_multidoc_diagram_exp"),
            DatasetSource("cosyn_multidoc_doc_exp"),
            DatasetSource("cosyn_multidoc_music_exp"),
            DatasetSource("cosyn_multidoc_table_exp"),
        ],
    ),
    SubMixture(
        "image_pointing",
        0.166,
        [
            DatasetSource(
                "pixmo_multi_points",
                root_size_factor=200_000,
                message_weight=0.2,
                override_p_high_res=0.30,
            ),
            DatasetSource(
                "pixmo_points_train",
                message_weight=0.2,
                override_p_high_res=0.30,
            ),
            DatasetSource(
                "pixmo_count_train",
                message_weight=0.2,
                override_p_high_res=0.30,
            ),
            DatasetSource(
                "pixmo_points_high_freq_train",
                message_weight=0.2,
                override_p_high_res=0.30,
            ),
            DatasetSource(
                "cosyn_point",
                message_weight=0.2,
                override_p_high_res=0.30,
            ),
        ],
    ),
    SubMixture("nlp", 0.166, [DatasetSource("tulu4")]),
]

ACADEMIC_MIXTURE_DATASETS: Tuple[str, ...] = tuple(
    src.name for src in IMAGE_ONLY_V9_SUBMIXTURES[1].datasets
)

# Gradual validation ladder for stage-2 training (expand until image-only-v9 is green).
VALIDATION_MIXTURES: Dict[str, Optional[Tuple[str, ...]]] = {
    "debug": DEBUG_MIXTURE_DATASETS,
    "demo": DEMO_MIXTURE_DATASETS,
    "pointing": POINTING_MIXTURE_DATASETS,
    "demo-pointing": DEMO_MIXTURE_DATASETS + POINTING_MIXTURE_DATASETS,
    "nlp-demo": NLP_MIXTURE_DATASETS + DEMO_MIXTURE_DATASETS,
    "academic": ACADEMIC_MIXTURE_DATASETS,
    "multi-image": MULTI_IMAGE_MIXTURE_DATASETS,
    "image-only-v9": None,
    # Pointing bisect (one source each).
    "pixmo_multi_points": ("pixmo_multi_points",),
    "pixmo_points_train": ("pixmo_points_train",),
    "pixmo_count_train": ("pixmo_count_train",),
    "pixmo_points_high_freq_train": ("pixmo_points_high_freq_train",),
    "cosyn_point": ("cosyn_point",),
}


def _source_lookup() -> Dict[str, DatasetSource]:
    out: Dict[str, DatasetSource] = {}
    for group in IMAGE_ONLY_V9_SUBMIXTURES:
        for src in group.datasets:
            out[src.name] = src
    return out


def get_image_only_v9_source(name: str) -> Optional[DatasetSource]:
    """Return mixture metadata for a dataset name, if it is in image-only-v9."""
    return _source_lookup().get(name)


def build_image_only_v9_dataset(
    name: str,
    tokenizer,
    seed: int = 0,
    *,
    max_sequence_length: Optional[int] = None,
):
    """Build a single image-only-v9 dataset by mm_olmo name."""
    sources = _source_lookup()
    if name not in sources:
        raise KeyError(f"Unknown image-only-v9 dataset: {name}")

    if name == "pixmo_ask_model_anything":
        return PixMoAmaDatasetConfig(seed=seed).build(tokenizer)
    if name == "pixmo_cap":
        cap_kw: Dict[str, Any] = dict(
            dataset_path=f"{PIXMO_DATASETS}/cap",
            mode="sft_demo",
            seed=seed,
        )
        if max_sequence_length is not None:
            cap_kw["max_sequence_length"] = max_sequence_length
        return PixMoCapDatasetConfig(**cap_kw).build(tokenizer)
    if name == "pixmo_cap_qa_as_user_qa":
        return PixMoCapQaDatasetConfig(seed=seed).build(tokenizer)
    if name == "tulu4":
        tulu_kw: Dict[str, Any] = dict(seed=seed)
        if max_sequence_length is not None:
            tulu_kw["max_sequence_length"] = max_sequence_length
        return Tulu4DatasetConfig(**tulu_kw).build(tokenizer)

    src = sources[name]
    if name == "correction_qa_multi_only_max5":
        from olmo_core.data.multimodal.multi_image_datasets import (
            CorrectionQaDatasetConfig,
        )

        return CorrectionQaDatasetConfig(seed=seed).build(tokenizer)
    if name.startswith("mantis_instruct_"):
        from olmo_core.data.multimodal.multi_image_datasets import (
            MantisInstructDatasetConfig,
        )

        subset = name[len("mantis_instruct_") :].replace("_multi_only", "")
        return MantisInstructDatasetConfig(subset=subset, seed=seed).build(tokenizer)
    if name.startswith("cosyn_multidoc_"):
        from olmo_core.data.multimodal.multi_image_datasets import (
            CoSynMultiDocDatasetConfig,
        )

        doc_type = name[len("cosyn_multidoc_") :].replace("_exp", "")
        return CoSynMultiDocDatasetConfig(
            doc_type=doc_type, use_exp=name.endswith("_exp"), seed=seed
        ).build(tokenizer)
    if name == "pixmo_multi_points":
        from olmo_core.data.multimodal.multi_image_datasets import (
            PixMoMultiPointsDatasetConfig,
        )

        return PixMoMultiPointsDatasetConfig(
            loss_token_weighting="none",
            message_weight=src.message_weight,
            p_high_res=src.override_p_high_res or 0.0,
            seed=seed,
        ).build(tokenizer)
    if name == "pixmo_points_train":
        return PixMoPointsDatasetConfig(
            kind="basic",
            loss_token_weighting="none",
            message_weight=src.message_weight,
            p_high_res=src.override_p_high_res or 0.0,
            seed=seed,
        ).build(tokenizer)
    if name == "pixmo_points_high_freq_train":
        return PixMoPointsDatasetConfig(
            kind="high_frequency",
            loss_token_weighting="none",
            message_weight=src.message_weight,
            p_high_res=src.override_p_high_res or 0.0,
            seed=seed,
        ).build(tokenizer)
    if name == "pixmo_count_train":
        return PixMoCountDatasetConfig(
            loss_token_weighting="none",
            message_weight=src.message_weight,
            p_high_res=src.override_p_high_res or 0.0,
            seed=seed,
        ).build(tokenizer)
    if name == "cosyn_point":
        return CoSynPointDatasetConfig(
            loss_token_weighting="none",
            message_weight=src.message_weight,
            p_high_res=src.override_p_high_res or 0.0,
            seed=seed,
        ).build(tokenizer)

    return AcademicDatasetConfig(
        name=name,
        loss_token_weighting="root_subsegments_root_tokens",
        message_weight=src.message_weight,
        seed=seed,
    ).build(tokenizer)


class _LazyDatasetMap:
    """Lazy dataset registry: builds each dataset on first access."""

    def __init__(
        self,
        tokenizer,
        seed: int,
        *,
        max_sequence_length: Optional[int] = None,
    ):
        self._tokenizer = tokenizer
        self._seed = seed
        self._max_sequence_length = max_sequence_length
        self._cache: Dict[str, object] = {}

    def keys(self):
        return _source_lookup().keys()

    def __contains__(self, name: str) -> bool:
        return name in _source_lookup()

    def __getitem__(self, name: str):
        if name not in self._cache:
            self._cache[name] = build_image_only_v9_dataset(
                name,
                self._tokenizer,
                self._seed,
                max_sequence_length=self._max_sequence_length,
            )
        return self._cache[name]


def build_image_only_v9_datasets(
    tokenizer,
    seed: int = 0,
    *,
    max_sequence_length: Optional[int] = None,
) -> _LazyDatasetMap:
    """Lazy registry of all 32 image-only-v9 datasets keyed by mm_olmo name."""
    return _LazyDatasetMap(tokenizer, seed, max_sequence_length=max_sequence_length)


def build_image_only_v9_mixture(
    tokenizer,
    seed: int = 0,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    max_sequence_length: int = 16384,
) -> Tuple[List, List[float]]:
    """Build weighted datasets for :class:`~olmo_core.data.multimodal.MixtureDataLoader`.

    Flattens ``IMAGE_ONLY_V9_SUBMIXTURES`` with mm_olmo SubMixture rate math, optionally
    restricting to ``dataset_names`` (weights are renormalized over the subset).
    """
    datasets_map = build_image_only_v9_datasets(
        tokenizer, seed, max_sequence_length=max_sequence_length
    )
    lengths = {name: len(datasets_map[name]) for name in datasets_map.keys()}
    flat = compute_flat_mixture_weights(IMAGE_ONLY_V9_SUBMIXTURES, lengths)

    if dataset_names is not None:
        allowed = set(dataset_names)
        flat = [(name, w) for name, w in flat if name in allowed]
        if not flat:
            raise ValueError(f"No mixture sources matched dataset_names={dataset_names!r}")
        norm = sum(w for _, w in flat)
        flat = [(name, w / norm) for name, w in flat]

    out_datasets = [datasets_map[name] for name, _ in flat]
    out_weights = [w for _, w in flat]
    return out_datasets, out_weights
