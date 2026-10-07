"""Mixed text-and-vision midtraining through the shared internal experiment runner."""

import json
from dataclasses import dataclass, field, fields
from math import isfinite
from pathlib import Path
from typing import Any

from olmo_core.config import Config, DType, _clean_opts
from olmo_core.data import InstanceFilterConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url, normalize_path, resource_path
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.attention.kda import KimiDeltaAttentionConfig
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath, ExpertParallelSchedule
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import Molmo2TokenIds, MultimodalLMConfig
from olmo_core.optim import (
    CosWithWarmup,
    OptimGroupOverride,
    PerGroupScheduler,
    Scheduler,
    SchedulerUnits,
)
from olmo_core.optim.multimodal_optimizer import MultimodalOLMoDDPOptimizerConfig
from olmo_core.train import CheckpointerConfig, Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.multimodal import (
    InitializeMultimodalModelCallback,
    MultimodalBeakerCallback,
    MultimodalCheckpointerCallback,
    MultimodalMetricSaverCallback,
    MultimodalWandBCallback,
    RestoreMetricsCallback,
)
from olmo_core.train.common import DurationUnit
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
)

from .experiment import CliContext, ExperimentConfig
from .vision_alignment import _STAGE1_V3_POST_SETUP
from .vision_alignment import _build_launch as _build_alignment_launch
from .vision_alignment import (
    _check_text_lm_matches_checkpoint,
    _checkpoint_lm_config,
    _image_token_rows,
    _load_text_config,
    _restore_pretraining_router_lb,
    _sample_one_annotation,
    _uses_document_mode,
    parse_cli_args,
    run,
    text_train_settings,
)
from .vision_alignment_data import (
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    STAGE1_V3_LOSS_TARGETS,
    STAGE1_V3_MEAN_LOSS_WEIGHTS,
    build_stage1_v3_sources,
)
from .vision_midtraining_data import (
    DEFAULT_MIDTRAINING_ARTIFACT_ROOT,
    DEFAULT_VISUAL_EXAMPLE_WEIGHTS,
    DEFAULT_VISUAL_LOSS_SHARES,
    DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS,
    TEXT_SOURCE_NAME,
    build_visual_sources,
    loss_mass_targets,
)

_DOLMA2_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
_SOURCE_MIX_PATH = "src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
_LEGACY_MAX_TOKENS = 50_000_000_000
_ALIGNED_CONNECTOR_LR_DIVISOR = 2
_ALIGNED_VISION_LR_DIVISOR = 2
"""From an alignment checkpoint (trained connector and vision encoder): both at half the LM's LR."""
_FRESH_CONNECTOR_LR_SCALE = 10
_FRESH_VISION_LR_DIVISOR = 5
"""From the text LM (fresh connector, pretrained SigLIP): the connector at 10x the LM's LR, the
vision encoder at a fifth."""
_NUM_NODES = 2
"""Mixed midtraining runs on two eight-GPU nodes, as Rustin ran it."""


@dataclass
class MixedMidtrainingRecipeConfig(Config):
    """Inputs for a multimodal continuation of a joint-alignment checkpoint.

    Component-level overrides are applied after constructing the defaults. The token
    budget sizes both the text mixture and the learning-rate schedule; a shorter trainer
    duration or hard stop does not shorten either of those horizons.
    """

    parent_checkpoint: str | None = None
    """Vision-alignment checkpoint to continue from (model-only handoff): the endpoint of any
    alignment stage (bridge, perception or joint), so stage ablations can feed midtraining.
    Set this or ``pretraining_checkpoint``."""
    pretraining_checkpoint: str | None = None
    """Text LM checkpoint to start from without vision alignment: the vision encoder and the
    connector are initialized as alignment's bridge initializes them (pretrained SigLIP, a fresh
    connector, image-token embedding rows). Set this or ``parent_checkpoint``."""
    vision_model_id: str = "google/siglip2-so400m-patch14-384"
    """Vision encoder loaded when starting from ``pretraining_checkpoint``."""
    vision_revision: str = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    text_config: str | None = None
    """Path to the text team's resolved mid-training ``config.json``.

    When set, the text mixture, token budget, global batch, LM learning rate and schedule,
    optimizer and train-module settings, trainer bookkeeping and launch resources are inherited
    from it; the connector and vision learning rates scale with the LM's. Without it, the
    legacy s002 recipe applies (OLMo 3 32B midtraining mix, LM learning rate 1e-5, EP8).
    """
    text_loss_share: float = 0.9
    """Target text fraction of expected supervised-token loss mass; one is text-only."""
    visual_example_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VISUAL_EXAMPLE_WEIGHTS)
    )
    """Relative example weights within the unreserved portion of visual loss mass."""
    visual_loss_shares: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VISUAL_LOSS_SHARES)
    )
    """Reserved shares of aggregate visual loss mass, disjoint from example-weighted sources."""
    sequence_length: int = 8192
    max_tokens: int | None = None
    """Token-position budget, rounded up to a whole number of global batches. ``None`` uses the
    text config's budget, else 50B tokens."""
    max_crops: int = 8
    source_mix_path: str = _SOURCE_MIX_PATH
    text_dataset: NumpyFSLDatasetConfig | None = None
    """Optional explicit text dataset, required for a non-Dolma2 parent tokenizer."""
    alignment_artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT
    midtraining_artifact_root: str = DEFAULT_MIDTRAINING_ARTIFACT_ROOT
    output_root: str = (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/checkpoints"
    )
    work_dir: str = "/weka/oe-training-default/rustin/dataset-cache/mixed-midtraining"
    hf_cache_dir: str | None = "/weka/oe-training-default/rustin/hf-cache/hub"
    tokenizer_revision: str | None = None
    prefetch_workers: int | None = None
    """Per-rank threads that load and preprocess examples ahead of the GPU step. ``None`` uses the
    text config's ``data_loader.num_workers`` (else 8). A ``recipe`` field so it can be set where
    ``--data_loader.*`` overrides also reach the text team's own loader (scaling-ladders)."""
    document_mode: bool | None = None
    """How packed examples are isolated in the language model. ``None``: document boundaries
    for recurrent (KDA) LMs, with the FLA kernels that support them. ``False``: no boundaries, the
    text team's kernels (``kernel_fun``), packed examples separated only by their tokens as text
    pretraining packs documents."""
    prefetch_max_in_flight: int | None = None
    """Examples the loader may hold preprocessed ahead of consumption per rank. ``None`` is the
    loader's default, ``max(2 * prefetch_workers, 4)``, which is less than one rank batch of
    examples; a deeper queue lets the loader work through the GPU step."""
    visual_data: str = "midtraining"
    """Visual sources: ``midtraining`` (Rustin's eight groups) or ``stage1_v3`` (the Molmo2
    Stage-1 v3 mixture, as alignment's perception and joint use it: one sampled annotation per
    example, its calibrated means, and its loss shares within the visual share)."""


@dataclass
class MixedMidtrainingExperimentConfig(ExperimentConfig):
    """A mixed-midtraining experiment using the standard model, data, and trainer configs."""

    model: MultimodalLMConfig  # type: ignore[assignment]
    dataset: MultimodalMixtureConfig  # type: ignore[assignment]
    data_loader: MixtureDataLoaderConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    recipe: MixedMidtrainingRecipeConfig = field(default_factory=MixedMidtrainingRecipeConfig)
    pretraining_checkpoint: str | None = None
    """Original language-model ancestry, when recorded by the alignment parent."""
    alignment_phase: str | None = None
    """The alignment stage of the parent checkpoint, or ``None`` when midtraining starts from
    the text LM."""


def _read_checkpoint_config(checkpoint: str) -> dict[str, Any]:
    with resource_path(checkpoint, "config.json").open() as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise OLMoConfigurationError("Checkpoint config must be an object")
    return config


def _build_recipe(
    cli: CliContext, overrides: list[tuple[str, Any]]
) -> MixedMidtrainingRecipeConfig:
    # Dataclass decoding coerces booleans to numbers; reject those before the ordinary merge.
    numeric_fields = {
        "recipe.text_loss_share",
        "recipe.sequence_length",
        "recipe.max_tokens",
        "recipe.max_crops",
    }
    numeric_maps = {
        "recipe.visual_example_weights",
        "recipe.visual_loss_shares",
        "dataset.mean_loss_weight",
    }
    for name, parsed in overrides:
        if name == "recipe":
            raise OLMoConfigurationError("Use dotted --recipe.FIELD overrides")
        if name in numeric_fields and isinstance(parsed, bool):
            raise OLMoConfigurationError(f"{name} must be numeric, not a boolean")
        for mapping in numeric_maps:
            values: tuple[Any, ...] = ()
            if name == mapping and isinstance(parsed, dict):
                values = tuple(parsed.values())
            elif name.startswith(mapping + "."):
                values = (parsed,)
            elif (
                name == "dataset"
                and mapping == "dataset.mean_loss_weight"
                and isinstance(parsed, dict)
            ):
                means = parsed.get("mean_loss_weight")
                if isinstance(means, dict):
                    values = tuple(means.values())
            if any(isinstance(item, bool) for item in values):
                raise OLMoConfigurationError(f"{mapping} values must be numeric, not booleans")
    recipe = MixedMidtrainingRecipeConfig().merge(cli.overrides, prefix="recipe")
    if bool(recipe.parent_checkpoint) == bool(recipe.pretraining_checkpoint):
        raise OLMoConfigurationError(
            "Set exactly one of recipe.parent_checkpoint (an alignment checkpoint) and "
            "recipe.pretraining_checkpoint (a text LM checkpoint, without alignment)"
        )
    for name, minimum in (("sequence_length", 2), ("max_tokens", 1), ("max_crops", 1)):
        value = getattr(recipe, name)
        if name == "max_tokens" and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise OLMoConfigurationError(f"recipe.{name} must be an integer of at least {minimum}")
    if recipe.visual_data not in ("midtraining", "stage1_v3"):
        raise OLMoConfigurationError("recipe.visual_data must be midtraining or stage1_v3")
    if (
        isinstance(recipe.text_loss_share, bool)
        or not isfinite(recipe.text_loss_share)
        or not 0 < recipe.text_loss_share <= 1
    ):
        raise OLMoConfigurationError("recipe.text_loss_share must be finite and in (0, 1]")
    return recipe


_ALIGNMENT_PHASES = ("bridge", "perception", "joint")


def _resolve_parent(
    recipe: MixedMidtrainingRecipeConfig,
) -> tuple[dict[str, Any], str | None, str]:
    assert recipe.parent_checkpoint is not None
    parent = _read_checkpoint_config(recipe.parent_checkpoint)
    phase = (parent.get("recipe") or {}).get(
        "phase", (parent.get("vision_alignment") or {}).get("phase", parent.get("phase"))
    )
    if phase not in _ALIGNMENT_PHASES:
        raise OLMoConfigurationError(
            f"recipe.parent_checkpoint must be a vision-alignment checkpoint, got phase {phase!r}"
        )
    if not isinstance(parent.get("model"), dict) or "lm" not in parent["model"]:
        raise OLMoConfigurationError("Alignment parent must record a multimodal model")
    ancestry = parent.get("pretraining_checkpoint") or (parent.get("artifacts") or {}).get(
        "base_checkpoint"
    )
    return parent, ancestry, phase


def _pretrained_lm(text: dict | None, ancestry: str | None) -> OLMoDDPModelConfig:
    """The text LM's own config: the text config's, else the pretraining checkpoint's."""
    if text is not None:
        lm = OLMoDDPModelConfig.from_dict(text["model"])
    elif ancestry:
        lm = OLMoDDPModelConfig.from_dict(_checkpoint_lm_config(ancestry))
    else:
        raise OLMoConfigurationError(
            "Restoring the LM's router settings needs recipe.text_config or a recorded "
            "pretraining checkpoint"
        )
    if not isinstance(lm, OLMoDDPModelConfig):
        raise OLMoConfigurationError("Mixed midtraining requires an OLMoDDP language model")
    return lm


def _lm_tokenizer(text: dict | None, checkpoint: str) -> TokenizerConfig:
    raw = ((text or {}).get("dataset") or {}).get("tokenizer")
    if raw is None:
        raw = (_read_checkpoint_config(checkpoint).get("dataset") or {}).get("tokenizer")
    if raw is None:
        raise OLMoConfigurationError("The text LM checkpoint does not identify its tokenizer")
    return TokenizerConfig.from_dict(raw)


def _apply_document_mode(model: MultimodalLMConfig, document_mode: bool | None) -> None:
    """Force the wrapper's isolation mode and pick the matching KDA kernels: FLA for document
    boundaries, the text team's ``kernel_fun`` without them."""
    if document_mode is None:
        return
    model.document_mode = document_mode
    lm = model.lm
    for block in [lm.block, *(getattr(lm, "block_overrides", None) or {}).values()]:
        mixer = getattr(block, "sequence_mixer", None)
        if isinstance(mixer, KimiDeltaAttentionConfig):
            mixer.use_experimental_kernels = not document_mode


def _build_lm_model(
    recipe: MixedMidtrainingRecipeConfig, text: dict | None, token_ids: Molmo2TokenIds
) -> MultimodalLMConfig:
    """The multimodal model around the text LM, as alignment's bridge builds it."""
    assert recipe.pretraining_checkpoint is not None
    lm_dict = (
        text["model"] if text is not None else _checkpoint_lm_config(recipe.pretraining_checkpoint)
    )
    lm = OLMoDDPModelConfig.from_dict(lm_dict)
    if not isinstance(lm, OLMoDDPModelConfig):
        raise OLMoConfigurationError("Mixed midtraining requires an OLMoDDP language model")
    if text is not None:
        _check_text_lm_matches_checkpoint(lm, recipe.pretraining_checkpoint)
    if _uses_document_mode(lm):
        # As in alignment: kernel_fun's KDA does not support packed documents (cu_seqlens).
        for block in [lm.block, *(lm.block_overrides or {}).values()]:
            mixer = getattr(block, "sequence_mixer", None)
            if isinstance(mixer, KimiDeltaAttentionConfig):
                mixer.use_experimental_kernels = False
    lm.two_batch_overlap = False
    model = MultimodalLMConfig.molmo2_vision_stack(lm, image_patch_token_id=token_ids.im_patch_id)
    _apply_document_mode(model, recipe.document_mode)
    return model


def _parent_tokenizer(parent: dict[str, Any], ancestry: str | None) -> TokenizerConfig:
    dataset = parent.get("dataset") or {}
    raw = dataset.get("tokenizer")
    if raw is None:
        raw = ((parent.get("text_dataset") or {}).get("dataset") or {}).get("tokenizer")
    if raw is None and ancestry:
        raw = (_read_checkpoint_config(ancestry).get("dataset") or {}).get("tokenizer")
    if raw is None:
        raise OLMoConfigurationError("Alignment parent does not identify its text tokenizer")
    tokenizer = TokenizerConfig.from_dict(raw)
    identifier = (parent.get("artifacts") or {}).get("tokenizer_id", tokenizer.identifier)
    if identifier != tokenizer.identifier:
        raise OLMoConfigurationError("Parent tokenizer identity differs from its text ancestry")
    return tokenizer


def _tokenizer_revision(recipe: MixedMidtrainingRecipeConfig, parent: dict[str, Any]) -> str | None:
    dataset, artifacts = parent.get("dataset") or {}, parent.get("artifacts") or {}
    revision = dataset.get("tokenizer_revision", artifacts.get("tokenizer_revision"))
    if revision is not None and recipe.tokenizer_revision not in (None, revision):
        raise OLMoConfigurationError("Requested tokenizer revision differs from alignment parent")
    return revision if revision is not None else recipe.tokenizer_revision


def _build_model(parent: dict[str, Any], text: dict | None = None) -> MultimodalLMConfig:
    model = MultimodalLMConfig.from_dict(parent["model"])
    if not isinstance(model.lm, OLMoDDPModelConfig):
        raise OLMoConfigurationError("Mixed midtraining requires an OLMoDDP language model")
    for block in [model.lm.block, *(model.lm.block_overrides or {}).values()]:
        if not isinstance(block, OLMoDDPTransformerBlockConfig):
            raise OLMoConfigurationError("Mixed midtraining requires OLMoDDP transformer blocks")
        if text is None:
            if isinstance(block.sequence_mixer, AttentionConfig):
                block.sequence_mixer.backend = AttentionBackendName.flex
            if block.ep is not None:
                block.ep.path = ExpertParallelPath.rowwise_nvshmem
                block.ep.schedule = ExpertParallelSchedule.normal
    if text is None:
        model.lm.recompute_each_block = True
        model.lm.recompute_all_blocks_by_chunk = False
    # With a text config, the parent's LM already carries the text run's runtime (alignment
    # built it from the same config). The wrapper feeds embeddings, which two-batch overlap
    # cannot take.
    model.lm.two_batch_overlap = False
    return model


def _build_text_dataset(
    recipe: MixedMidtrainingRecipeConfig,
    tokenizer: TokenizerConfig,
    budget: int,
    batch_size: int,
    text: dict | None = None,
) -> NumpyFSLDatasetConfig:
    if recipe.text_dataset is not None:
        dataset = recipe.text_dataset.copy()
    elif text is not None:
        # The text team's mid-training mixture; the mixture cache lives in this recipe's own
        # work directory.
        dataset = NumpyFSLDatasetConfig.from_dict(text["dataset"])
        dataset.work_dir = recipe.work_dir
    else:
        if tokenizer != TokenizerConfig.dolma2():
            raise OLMoConfigurationError(
                "The default text mixture uses Dolma2; supply a compatible recipe.text_dataset"
            )
        source_list = SourceMixtureList.from_file(recipe.source_mix_path)
        source_list.validate()
        dataset = NumpyFSLDatasetConfig.from_src_mix(
            SourceMixtureDatasetConfig(
                source_list=source_list,
                requested_tokens=budget,
                global_batch_size=batch_size,
                processes=16,
                seed=1337,
            ),
            tokenizer=tokenizer.copy(),
            sequence_length=recipe.sequence_length,
            max_target_sequence_length=recipe.sequence_length,
            work_dir=recipe.work_dir,
            instance_filter_config=InstanceFilterConfig(
                repetition_min_period=1, repetition_max_period=13, repetition_max_count=32
            ),
        )
    if type(dataset) is not NumpyFSLDatasetConfig or dataset.tokenizer != tokenizer:
        raise OLMoConfigurationError("Midtraining text must use the alignment parent's tokenizer")
    if dataset.sequence_length != recipe.sequence_length:
        raise OLMoConfigurationError("Text and recipe sequence lengths must agree")
    if dataset.source_mixture_config is not None:
        dataset.source_mixture_config.requested_tokens = budget
        dataset.source_mixture_config.global_batch_size = batch_size
    dataset.validate()
    return dataset


def _build_data_loader(
    cli: CliContext, recipe: MixedMidtrainingRecipeConfig, text: dict | None = None
) -> MixtureDataLoaderConfig:
    batch_size, workers = 128 * recipe.sequence_length, 8
    if text is not None:
        batch_size = int(text["data_loader"]["global_batch_size"])
        workers = int(text["data_loader"].get("num_workers", workers))
    if recipe.prefetch_workers is not None:
        workers = recipe.prefetch_workers
    return MixtureDataLoaderConfig(
        global_batch_size=batch_size,
        sequence_length=recipe.sequence_length,
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        seed=95818,
        pack=True,
        pack_buffer_size=48,
        pack_max_crops=64,
        pack_image_weight=1.0,
        # Exact resume from the checkpointed cursor and the collator metadata the train module
        # reads, as in alignment.
        continuous_stream=True,
        batch_metadata=True,
        # Half the pixel bytes copied to the device; the bf16 tower sees the same values.
        image_dtype=DType.bfloat16,
        # Only the real crops are collated and copied (no padding to the batch's crop maximum).
        compact_images=True,
        prefetch_workers=workers,
        prefetch_max_in_flight=recipe.prefetch_max_in_flight,
        max_consecutive_data_errors=0,
        max_total_data_errors=0,
    ).merge(cli.overrides, prefix="data_loader")


def _build_train_module(
    sequence_length: int,
    budget: int,
    batch_size: int,
    *,
    text_only: bool,
    text: dict | None = None,
    from_alignment: bool = True,
) -> MultimodalOLMoDDPTrainModuleConfig:
    optim_settings, module_settings = text_train_settings(text)
    scheduler: Scheduler
    if text is not None:
        text_optim = text["train_module"]["optim"]
        lm_lr = float(text_optim["lr"])
        if from_alignment:
            connector_lr = lm_lr / _ALIGNED_CONNECTOR_LR_DIVISOR
            vision_lr = lm_lr / _ALIGNED_VISION_LR_DIVISOR
        else:
            connector_lr = lm_lr * _FRESH_CONNECTOR_LR_SCALE
            vision_lr = lm_lr / _FRESH_VISION_LR_DIVISOR
        # Weight decay as the text config sets it for every parameter (David's 0.1).
        component_decay: dict[str, Any] = {}
        scheduler = Scheduler.from_dict(text["train_module"]["scheduler"])
        lm_groups = [
            OptimGroupOverride.from_dict(group) for group in text_optim.get("group_overrides") or []
        ]
    else:
        lm_lr, connector_lr, vision_lr = 1e-5, 2e-5, 1e-6
        component_decay = {"weight_decay": 0.0}
        scheduler = CosWithWarmup(
            warmup=200 * batch_size, alpha_f=0.1, t_max=budget, units=SchedulerUnits.tokens
        )
        optim_settings.update(eps=1e-8, weight_decay=0.1)
        lm_groups = [
            OptimGroupOverride(
                params=[
                    "*lm.embeddings.weight",
                    "*lm.embedding_norm.*",
                    "*lm.blocks.*norm*.weight",
                    "*lm.lm_head.norm.*",
                ],
                opts={"weight_decay": 0.0},
            )
        ]
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=2 * sequence_length,
        max_sequence_length=sequence_length,
        optim=MultimodalOLMoDDPOptimizerConfig(
            lr=lm_lr,
            group_overrides=[
                OptimGroupOverride(
                    params=["*connector.*"],
                    opts={"lr": connector_lr, **component_decay, "scheduler_name": "connector"},
                ),
                OptimGroupOverride(
                    params=["*vision.*"],
                    opts={
                        "lr": 0.0 if text_only else vision_lr,
                        **component_decay,
                        "scheduler_name": "vision",
                    },
                ),
                *lm_groups,
            ],
            foreach_chunk_size=50_000_000,
            clip_grad_norm_by_scheduler_group=True,
            **optim_settings,
        ),
        freeze_params=["vision.*"] if text_only else [],
        train_embedding_rows=None,
        vision_activation_checkpointing=not text_only,
        # Off as in alignment: the checkpoint wrapper breaks the connector's reset_parameters
        # under the OLMoDDP init order, and its activations are a negligible share of memory.
        connector_activation_checkpointing=False,
        response_logits_only=True,
        diagnostics_interval=100,
        scheduler=PerGroupScheduler(
            schedulers={"connector": scheduler.copy(), "vision": scheduler.copy()},
            default=scheduler,
        ),
        **module_settings,
    )


def _build_trainer(
    cli: CliContext,
    recipe: MixedMidtrainingRecipeConfig,
    budget: int,
    text: dict | None = None,
    token_ids: Molmo2TokenIds | None = None,
) -> TrainerConfig:
    bookkeeping: dict[str, Any] = dict(
        checkpointer=CheckpointerConfig(load_thread_count=8),
        metrics_collect_interval=5,
        cancel_check_interval=5,
    )
    inherited_callbacks: dict[str, Any] = {
        "gpu_monitor": GPUMemoryMonitorCallback(),
        "config_saver": ConfigSaverCallback(),
        "garbage_collector": GarbageCollectorCallback(),
    }
    checkpointer = MultimodalCheckpointerCallback(
        save_interval=10_000, ephemeral_save_interval=500, save_async=False, max_checkpoints=2
    )
    if text is not None:
        # Bookkeeping, checkpoint cadence and callbacks come from the text run; its W&B, Beaker
        # and notifier callbacks are replaced by the multimodal ones below.
        text_trainer = TrainerConfig.from_dict(text["trainer"])
        bookkeeping = dict(
            checkpointer=text_trainer.checkpointer,
            metrics_collect_interval=text_trainer.metrics_collect_interval,
            cancel_check_interval=text_trainer.cancel_check_interval,
            async_bookkeeping=text_trainer.async_bookkeeping,
            bookkeeping_soft_timeout=text_trainer.bookkeeping_soft_timeout,
        )
        inherited_callbacks = {
            name: callback
            for name, callback in text_trainer.callbacks.items()
            if name not in ("checkpointer", "wandb", "slack_notifier", "beaker")
        }
        text_checkpointer = text_trainer.callbacks.get("checkpointer")
        if isinstance(text_checkpointer, CheckpointerCallback):
            checkpointer = MultimodalCheckpointerCallback(
                **{
                    f.name: getattr(text_checkpointer, f.name)
                    for f in fields(CheckpointerCallback)
                    if f.init and not f.name.startswith("_")
                }
            )
    trainer = TrainerConfig(
        save_folder=f"{recipe.output_root}/{cli.run_name}",
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        save_overwrite=False,
        # From an alignment checkpoint: model-only handoff. From the text LM: the initialization
        # callback loads it (and the vision encoder); resume uses the run's own checkpoints.
        load_path=recipe.parent_checkpoint,
        load_strategy=LoadStrategy.always
        if recipe.parent_checkpoint
        else LoadStrategy.if_available,
        load_optim_state=False if recipe.parent_checkpoint else None,
        load_trainer_state=False if recipe.parent_checkpoint else None,
        max_duration=Duration.tokens(budget),
        **bookkeeping,
    )
    for name, callback in inherited_callbacks.items():
        trainer = trainer.with_callback(name, callback)
    if recipe.pretraining_checkpoint:
        assert token_ids is not None
        trainer = trainer.with_callback(
            "initialize_multimodal",
            InitializeMultimodalModelCallback(
                language_checkpoint=recipe.pretraining_checkpoint,
                vision_model_id=recipe.vision_model_id,
                vision_revision=recipe.vision_revision,
                cache_dir=recipe.hf_cache_dir,
                image_token_ids=_image_token_rows(token_ids),
                seed=6198,
            ),
        )
    return (
        trainer.with_callback("checkpointer", checkpointer)
        .with_callback("beaker", MultimodalBeakerCallback())
        .with_callback(
            "wandb",
            MultimodalWandBCallback(
                name=cli.run_name, project="mixed-midtraining", auto_resume=True
            ),
        )
        .with_callback(
            "metrics",
            MultimodalMetricSaverCallback(
                save_interval=5, final_metrics_fname="metrics-final.json"
            ),
        )
        .with_callback("restore_metrics", RestoreMetricsCallback(metrics_callback="metrics"))
    )


def _build_launch(
    cli: CliContext,
    *,
    work_dir: str = MixedMidtrainingRecipeConfig.work_dir,
    text: dict | None = None,
) -> BeakerLaunchConfig | None:
    # The alignment launcher: workspace, budget, secrets, and (with a text config) the text
    # run's image, install step, resources and environment.
    launch = _build_alignment_launch(cli, work_dir=work_dir, text=text)
    if launch is not None:
        launch.num_nodes = _NUM_NODES
    return launch


def _explicit_mean(overrides: list[tuple[str, Any]], name: str) -> bool:
    for key, value in overrides:
        if key in {f"dataset.mean_loss_weight.{name}", "dataset.mean_loss_weight"}:
            return True
        if (
            key == "dataset"
            and isinstance(value, dict)
            and name in (value.get("mean_loss_weight") or {})
        ):
            return True
    return False


def _configure_loss(
    config: MixedMidtrainingExperimentConfig,
    overrides: list[tuple[str, Any]],
    text: NumpyFSLDatasetConfig,
    initial_visual_sources: dict[str, Config],
    reusable_visual_calibration: bool,
) -> None:
    recipe, dataset = config.recipe, config.dataset
    if text.label_mask_paths is None:
        expected_mean = float(recipe.sequence_length - 1)
        if dataset.mean_loss_weight.get(TEXT_SOURCE_NAME, expected_mean) != expected_mean:
            raise OLMoConfigurationError(
                "Unmasked text mean loss weight must equal sequence_length - 1"
            )
        dataset.mean_loss_weight[TEXT_SOURCE_NAME] = expected_mean
    elif recipe.text_loss_share == 1.0:
        # A singleton source needs no relative calibration, including for masked text.
        dataset.mean_loss_weight[TEXT_SOURCE_NAME] = 1.0
    elif not _explicit_mean(overrides, TEXT_SOURCE_NAME):
        raise OLMoConfigurationError(
            "Masked text requires explicit dataset.mean_loss_weight.text_midtraining"
        )
    if recipe.text_loss_share == 1.0:
        if set(dataset.sources) != {TEXT_SOURCE_NAME}:
            raise OLMoConfigurationError("Text-only midtraining must not include visual sources")
        dataset.mean_loss_weight = {TEXT_SOURCE_NAME: dataset.mean_loss_weight[TEXT_SOURCE_NAME]}
    else:
        changed = [
            name
            for name, source in dataset.sources.items()
            if name != TEXT_SOURCE_NAME
            and (not reusable_visual_calibration or source != initial_visual_sources.get(name))
            and not _explicit_mean(overrides, name)
        ]
        if changed:
            raise OLMoConfigurationError(
                f"Supply calibrated dataset.mean_loss_weight for changed visual sources: {changed}"
            )

        def validate_length(component: Config):
            if (
                getattr(component, "max_sequence_length", recipe.sequence_length)
                != recipe.sequence_length
            ):
                raise OLMoConfigurationError("Visual and text sequence lengths must agree")

        for name, source in dataset.sources.items():
            if name != TEXT_SOURCE_NAME:
                source.apply(validate_length)
    targets = loss_mass_targets(
        dataset.mean_loss_weight,
        target_text_loss_mass=recipe.text_loss_share,
        visual_example_weights=recipe.visual_example_weights,
        visual_loss_shares=recipe.visual_loss_shares,
    )
    if (
        any(
            name.startswith("dataset.target_loss_mass")
            or (name == "dataset" and isinstance(value, dict) and "target_loss_mass" in value)
            for name, value in overrides
        )
        and dataset.target_loss_mass != targets
    ):
        raise OLMoConfigurationError(
            "Set recipe.text_loss_share to change the supervised-loss allocation"
        )
    dataset.target_loss_mass = targets
    dataset.sampling_weights()
    config.train_module.source_loss_mass_targets = dict(targets)


def _validate_config(
    config: MixedMidtrainingExperimentConfig,
    overrides: list[tuple[str, Any]],
    tokenizer: TokenizerConfig,
    revision: str | None,
    budget: int,
    ancestry: str | None,
    initial_visual_sources: dict[str, Config],
    reusable_visual_calibration: bool,
) -> None:
    recipe, dataset, loader = config.recipe, config.dataset, config.data_loader
    if dataset.tokenizer != tokenizer or dataset.tokenizer_revision != revision:
        raise OLMoConfigurationError("Training tokenizer and revision must match alignment parent")
    if (
        loader.sequence_length != recipe.sequence_length
        or config.train_module.max_sequence_length != recipe.sequence_length
    ):
        raise OLMoConfigurationError(
            "Recipe, data-loader, and train-module sequence lengths must agree"
        )
    if (
        type(config.train_module.rank_microbatch_size) is not int
        or config.train_module.rank_microbatch_size <= 0
        or config.train_module.rank_microbatch_size % recipe.sequence_length
        or loader.global_batch_size % config.train_module.rank_microbatch_size
    ):
        raise OLMoConfigurationError("Global and microbatches must contain whole sequences")
    if loader.source_groups is not None or loader.group_sequence_quotas is not None:
        raise OLMoConfigurationError(
            "Fixed sequence quotas cannot implement calibrated loss shares"
        )
    replay = dataset.sources.get(TEXT_SOURCE_NAME)
    if not isinstance(replay, PretrainingReplayConfig) or replay.split != "all":
        raise OLMoConfigurationError("Mixed midtraining requires the complete explicit text replay")
    text = replay.resolve_dataset()
    if text.tokenizer != tokenizer or text.sequence_length != recipe.sequence_length:
        raise OLMoConfigurationError(
            "Text source tokenizer and sequence length must match training"
        )
    if text.source_mixture_config is not None and (
        text.source_mixture_config.requested_tokens != budget
        or text.source_mixture_config.global_batch_size != loader.global_batch_size
    ):
        raise OLMoConfigurationError("Text allocation must retain the complete recipe token budget")
    _configure_loss(config, overrides, text, initial_visual_sources, reusable_visual_calibration)
    if dataset.model_vocab_size != config.model.lm.vocab_size:
        raise OLMoConfigurationError("Dataset and model vocabulary sizes must agree")
    if config.model.connector.output_dim != config.model.lm.d_model:
        raise OLMoConfigurationError("Connector output width must match the language model")
    _, token_ids = dataset.build_tokenizer()
    if config.model.image_patch_token_id != token_ids.im_patch_id:
        raise OLMoConfigurationError("Model image token ID must match the alignment tokenizer")
    if config.trainer.no_checkpoints:
        raise OLMoConfigurationError(
            "Mixed midtraining requires checkpoint loading; trainer.no_checkpoints must be false. "
            "Use trainer.callbacks.checkpointer.enabled=false to disable checkpoint writes."
        )
    if config.trainer.load_path != recipe.parent_checkpoint:
        raise OLMoConfigurationError(
            "Use recipe.parent_checkpoint or recipe.pretraining_checkpoint to select the initial model"
        )
    if recipe.parent_checkpoint and (
        config.trainer.load_strategy != LoadStrategy.always
        or config.trainer.load_optim_state is not False
        or config.trainer.load_trainer_state is not False
    ):
        raise OLMoConfigurationError("The alignment handoff requires model-only checkpoint loading")
    if recipe.pretraining_checkpoint and "initialize_multimodal" not in config.trainer.callbacks:
        raise OLMoConfigurationError(
            "Starting from the text LM requires the initialization callback"
        )
    if config.pretraining_checkpoint != ancestry:
        raise OLMoConfigurationError("Pretraining ancestry must match the starting checkpoint")
    source = recipe.parent_checkpoint or recipe.pretraining_checkpoint
    assert source is not None
    paths = []
    for value in (source, config.trainer.save_folder):
        path = normalize_path(value).rstrip("/")
        paths.append(path if is_url(path) else str(Path(path).resolve()))
    source_path, output_path = paths
    if (
        source_path == output_path
        or source_path.startswith(output_path + "/")
        or output_path.startswith(source_path + "/")
    ):
        raise OLMoConfigurationError("Use a separate output folder for mixed midtraining")


def build_config(
    cli: CliContext,
    text_config: ExperimentConfig | dict[str, Any] | None = None,
    launch: BeakerLaunchConfig | None = None,
) -> MixedMidtrainingExperimentConfig:
    """Build mixed midtraining from checkpoint metadata and ordinary component overrides.

    No token arrays or visual datasets are opened. Dataset preparation and checkpoint
    loading remain the responsibility of the standard experiment runner.

    :param cli: The command line.
    :param text_config: The text team's mid-training config, built by their own launcher (for
        example a scaling-ladders midtraining or microanneal workload), in place of
        ``recipe.text_config``. Either an experiment config or its ``as_config_dict()``.
    :param launch: That launcher's own Beaker launch config (image, secrets, environment), used
        on two nodes in place of the one this recipe builds.
    """
    overrides = _clean_opts(cli.overrides)
    recipe = _build_recipe(cli, overrides)
    if text_config is not None and recipe.text_config:
        raise OLMoConfigurationError("Pass the text config in memory or as recipe.text_config")
    text: dict[str, Any] | None = None
    if isinstance(text_config, ExperimentConfig):
        text = text_config.as_config_dict()
    elif text_config is not None:
        text = dict(text_config)
    elif recipe.text_config:
        text = _load_text_config(recipe.text_config)
    if text is not None:
        for section in ("model", "train_module", "trainer", "data_loader"):
            if section not in text:
                raise OLMoConfigurationError(f"The text config lacks the {section!r} section")
    phase: str | None = None
    if recipe.parent_checkpoint:
        parent, ancestry, phase = _resolve_parent(recipe)
        tokenizer = _parent_tokenizer(parent, ancestry)
        revision = _tokenizer_revision(recipe, parent)
    else:
        ancestry = recipe.pretraining_checkpoint
        assert ancestry is not None
        tokenizer = _lm_tokenizer(text, ancestry)
        revision = recipe.tokenizer_revision
        if revision is None and tokenizer == TokenizerConfig.dolma2():
            revision = _DOLMA2_REVISION
    loader = _build_data_loader(cli, recipe, text)
    if (
        type(loader.global_batch_size) is not int
        or loader.global_batch_size <= 0
        or loader.global_batch_size % recipe.sequence_length
    ):
        raise OLMoConfigurationError(
            "Global batch size must contain a positive whole number of sequences"
        )
    max_tokens = recipe.max_tokens
    if max_tokens is None:
        max_tokens = _LEGACY_MAX_TOKENS
        if text is not None:
            duration = text["trainer"]["max_duration"]
            unit = DurationUnit(duration["unit"])
            if unit == DurationUnit.tokens:
                max_tokens = int(duration["value"])
            elif unit == DurationUnit.steps:
                # As the microanneal workload sizes its mixture: steps x the global batch.
                max_tokens = int(duration["value"]) * loader.global_batch_size
            else:
                raise OLMoConfigurationError(
                    "The text config's duration must be in tokens or steps"
                )
    budget = (
        (max_tokens + loader.global_batch_size - 1) // loader.global_batch_size
    ) * loader.global_batch_size
    text_data = _build_text_dataset(recipe, tokenizer, budget, loader.global_batch_size, text)
    token_ids: Molmo2TokenIds | None = None
    if recipe.parent_checkpoint:
        model = _build_model(parent, text)
        _apply_document_mode(model, recipe.document_mode)
        if phase != "joint":
            # Bridge and perception freeze the LM and switch router load balancing off; the LM
            # trains here, so the text LM's own coefficients come back.
            assert isinstance(model.lm, OLMoDDPModelConfig)
            _restore_pretraining_router_lb(model.lm, _pretrained_lm(text, ancestry))
    else:
        _, token_ids = MultimodalMixtureConfig(
            tokenizer=tokenizer.copy(),
            tokenizer_revision=revision,
            tokenizer_cache_dir=recipe.hf_cache_dir,
        ).build_tokenizer()
        model = _build_lm_model(recipe, text, token_ids)
    stage1_v3 = recipe.visual_data == "stage1_v3" and recipe.text_loss_share < 1.0
    if stage1_v3:
        means = dict(STAGE1_V3_MEAN_LOSS_WEIGHTS)
        recipe.visual_example_weights = {}
        recipe.visual_loss_shares = dict(STAGE1_V3_LOSS_TARGETS)
    else:
        means = dict(DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS) if recipe.text_loss_share < 1.0 else {}
    if text_data.label_mask_paths is None:
        means[TEXT_SOURCE_NAME] = float(recipe.sequence_length - 1)
    elif recipe.text_loss_share == 1.0:
        means[TEXT_SOURCE_NAME] = 1.0
    replaced_sources = any(name in {"dataset", "dataset.sources"} for name, _ in overrides)
    visual_sources: dict[str, Config] = {}
    if stage1_v3 and not replaced_sources:
        visual_sources = build_stage1_v3_sources(
            "joint", recipe.sequence_length, recipe.alignment_artifact_root
        )
        # KDA's document mode isolates packed documents, not sibling annotation branches.
        _sample_one_annotation(visual_sources)
    elif recipe.text_loss_share < 1.0 and not replaced_sources:
        visual_sources = build_visual_sources(
            sequence_length=recipe.sequence_length,
            max_crops=recipe.max_crops,
            alignment_artifact_root=recipe.alignment_artifact_root,
            midtraining_artifact_root=recipe.midtraining_artifact_root,
        )
    dataset = MultimodalMixtureConfig(
        tokenizer=text_data.tokenizer.copy(),
        sources=dict(
            sorted(
                {
                    **visual_sources,
                    TEXT_SOURCE_NAME: PretrainingReplayConfig(
                        dataset=text_data.copy(), split="all"
                    ),
                }.items()
            )
        ),
        mean_loss_weight=means,
        tokenizer_revision=revision,
        tokenizer_cache_dir=recipe.hf_cache_dir,
        model_vocab_size=model.lm.vocab_size,
    )
    config = MixedMidtrainingExperimentConfig(
        run_name=cli.run_name,
        launch=(
            _build_launch(cli, work_dir=recipe.work_dir, text=text)
            if launch is None
            else launch.replace(num_nodes=_NUM_NODES)
        ),
        model=model,
        dataset=dataset,
        data_loader=loader,
        train_module=_build_train_module(
            recipe.sequence_length,
            budget,
            loader.global_batch_size,
            text_only=recipe.text_loss_share == 1.0,
            text=text,
            from_alignment=bool(recipe.parent_checkpoint),
        ),
        trainer=_build_trainer(cli, recipe, budget, text, token_ids),
        recipe=recipe,
        pretraining_checkpoint=ancestry,
        alignment_phase=phase,
        init_seed=6198,
    ).merge(cli.overrides)
    if stage1_v3 and config.launch is not None:
        # The v3 Stage-1 sources need packages beyond the text image (PDF rendering, HDF5), as
        # alignment installs them for its stage1_v3 data.
        config.launch.post_setup = " && ".join(
            step for step in (config.launch.post_setup, _STAGE1_V3_POST_SETUP) if step
        )
    reusable_visual_calibration = (
        stage1_v3
        and recipe.sequence_length == 8192
        and tokenizer == TokenizerConfig.dolma2()
        and revision == _DOLMA2_REVISION
    ) or (
        recipe.sequence_length == 8192
        and recipe.max_crops == 8
        and recipe.alignment_artifact_root == DEFAULT_ALIGNMENT_ARTIFACT_ROOT
        and recipe.midtraining_artifact_root == DEFAULT_MIDTRAINING_ARTIFACT_ROOT
        and tokenizer == TokenizerConfig.dolma2()
        and revision == _DOLMA2_REVISION
    )
    _validate_config(
        config,
        overrides,
        tokenizer,
        revision,
        budget,
        ancestry,
        visual_sources,
        reusable_visual_calibration,
    )
    return config


def main() -> None:
    """Build mixed midtraining from the command line and run ``launch``, ``train`` or ``dry_run``."""
    cli = parse_cli_args()
    config = build_config(cli)
    cli.cmd.prepare_environment(config)
    run(cli.cmd, config)  # type: ignore[arg-type]
