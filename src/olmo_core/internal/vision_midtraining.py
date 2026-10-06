"""Mixed text-and-vision midtraining through the shared internal experiment runner."""

import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from olmo_core.config import Config, DType, _clean_opts
from olmo_core.data import InstanceFilterConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url, normalize_path, resource_path
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerLaunchConfig
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath, ExpertParallelSchedule
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.optim import (
    CosWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    PerGroupScheduler,
    SchedulerUnits,
)
from olmo_core.train import CheckpointerConfig, Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    MetricSaverCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.restore_metrics import RestoreMetricsCallback
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)

from .common import build_launch_config
from .experiment import CliContext, ExperimentConfig
from .vision_alignment_data import DEFAULT_ALIGNMENT_ARTIFACT_ROOT
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


@dataclass
class MixedMidtrainingRecipeConfig(Config):
    """Inputs for a multimodal continuation of a joint-alignment checkpoint.

    Component-level overrides are applied after constructing the defaults. The token
    budget sizes both the text mixture and the learning-rate schedule; a shorter trainer
    duration or hard stop does not shorten either of those horizons.
    """

    parent_checkpoint: str | None = None
    """Joint-alignment checkpoint used for a fresh, model-only handoff."""
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
    max_tokens: int = 50_000_000_000
    """Token-position budget, rounded up to a whole number of global batches."""
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


@dataclass
class MixedMidtrainingExperimentConfig(ExperimentConfig):
    """A mixed-midtraining experiment using the standard model, data, and trainer configs."""

    model: MultimodalLMConfig
    dataset: MultimodalMixtureConfig
    data_loader: MixtureDataLoaderConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    recipe: MixedMidtrainingRecipeConfig = field(default_factory=MixedMidtrainingRecipeConfig)
    pretraining_checkpoint: str | None = None
    """Original language-model ancestry, when recorded by the alignment parent."""


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
    if not recipe.parent_checkpoint:
        raise OLMoConfigurationError("Set recipe.parent_checkpoint to a joint-alignment checkpoint")
    for name, minimum in (("sequence_length", 2), ("max_tokens", 1), ("max_crops", 1)):
        value = getattr(recipe, name)
        if type(value) is not int or value < minimum:
            raise OLMoConfigurationError(f"recipe.{name} must be an integer of at least {minimum}")
    if (
        isinstance(recipe.text_loss_share, bool)
        or not isfinite(recipe.text_loss_share)
        or not 0 < recipe.text_loss_share <= 1
    ):
        raise OLMoConfigurationError("recipe.text_loss_share must be finite and in (0, 1]")
    return recipe


def _resolve_parent(recipe: MixedMidtrainingRecipeConfig) -> tuple[dict[str, Any], str | None]:
    assert recipe.parent_checkpoint is not None
    parent = _read_checkpoint_config(recipe.parent_checkpoint)
    phase = (parent.get("recipe") or {}).get(
        "phase", (parent.get("vision_alignment") or {}).get("phase", parent.get("phase"))
    )
    if phase != "joint":
        raise OLMoConfigurationError(f"Mixed midtraining requires joint alignment, got {phase!r}")
    if not isinstance(parent.get("model"), dict) or "lm" not in parent["model"]:
        raise OLMoConfigurationError("Joint-alignment parent must record a multimodal model")
    ancestry = parent.get("pretraining_checkpoint") or (parent.get("artifacts") or {}).get(
        "base_checkpoint"
    )
    return parent, ancestry


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


def _build_model(parent: dict[str, Any]) -> MultimodalLMConfig:
    model = MultimodalLMConfig.from_dict(parent["model"])
    if not isinstance(model.lm, OLMoDDPModelConfig):
        raise OLMoConfigurationError("Mixed midtraining requires an OLMoDDP language model")
    for block in [model.lm.block, *(model.lm.block_overrides or {}).values()]:
        if not isinstance(block, OLMoDDPTransformerBlockConfig):
            raise OLMoConfigurationError("Mixed midtraining requires OLMoDDP transformer blocks")
        if isinstance(block.sequence_mixer, AttentionConfig):
            block.sequence_mixer.backend = AttentionBackendName.flex
        if block.ep is not None:
            block.ep.path = ExpertParallelPath.rowwise_nvshmem
            block.ep.schedule = ExpertParallelSchedule.normal
    model.lm.recompute_each_block = True
    model.lm.recompute_all_blocks_by_chunk = False
    model.lm.two_batch_overlap = False
    return model


def _build_text_dataset(
    recipe: MixedMidtrainingRecipeConfig, tokenizer: TokenizerConfig, budget: int, batch_size: int
) -> NumpyFSLDatasetConfig:
    if recipe.text_dataset is not None:
        dataset = recipe.text_dataset.copy()
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
    cli: CliContext, recipe: MixedMidtrainingRecipeConfig
) -> MixtureDataLoaderConfig:
    return MixtureDataLoaderConfig(
        global_batch_size=128 * recipe.sequence_length,
        sequence_length=recipe.sequence_length,
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        seed=95818,
        pack=True,
        pack_buffer_size=48,
        pack_max_crops=64,
        pack_image_weight=1.0,
        prefetch_workers=8,
        max_consecutive_data_errors=0,
        max_total_data_errors=0,
        text_only=recipe.text_loss_share == 1.0,
    ).merge(cli.overrides, prefix="data_loader")


def _build_train_module(
    sequence_length: int, budget: int, batch_size: int, *, text_only: bool
) -> MultimodalOLMoDDPTrainModuleConfig:
    scheduler = CosWithWarmup(
        warmup=200 * batch_size, alpha_f=0.1, t_max=budget, units=SchedulerUnits.tokens
    )
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=2 * sequence_length,
        max_sequence_length=sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=1e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            group_overrides=[
                OptimGroupOverride(
                    params=["*connector.*"],
                    opts={"lr": 2e-5, "weight_decay": 0.0, "scheduler_name": "connector"},
                ),
                OptimGroupOverride(
                    params=["*vision.*"],
                    opts={
                        "lr": 0.0 if text_only else 1e-6,
                        "weight_decay": 0.0,
                        "scheduler_name": "vision",
                    },
                ),
                OptimGroupOverride(
                    params=[
                        "*lm.embeddings.weight",
                        "*lm.embedding_norm.*",
                        "*lm.blocks.*norm*.weight",
                        "*lm.lm_head.norm.*",
                    ],
                    opts={"weight_decay": 0.0},
                ),
            ],
            compile=False,
            foreach_chunk_size=50_000_000,
            sigma_factor=12,
            max_grad_norm=1.0,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        freeze_params=["vision.*"] if text_only else [],
        train_embedding_rows=None,
        vision_activation_checkpointing=not text_only,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        diagnostics_interval=100,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=True,
        scheduler=PerGroupScheduler(
            schedulers={"connector": scheduler.copy(), "vision": scheduler.copy()},
            default=scheduler,
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_dtype=DType.float32,
            only_allreduce_last_microbatch=True,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=8),
    )


def _build_trainer(
    cli: CliContext, recipe: MixedMidtrainingRecipeConfig, budget: int
) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=f"{recipe.output_root}/{cli.run_name}",
            work_dir=f"{recipe.work_dir}/{cli.run_name}",
            save_overwrite=False,
            load_path=recipe.parent_checkpoint,
            load_strategy=LoadStrategy.always,
            load_optim_state=False,
            load_trainer_state=False,
            checkpointer=CheckpointerConfig(load_thread_count=8),
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.tokens(budget),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=500,
                save_async=False,
                max_checkpoints=2,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "wandb", WandBCallback(name=cli.run_name, project="mixed-midtraining", auto_resume=True)
        )
        .with_callback(
            "metrics",
            MetricSaverCallback(save_interval=5, final_metrics_fname="metrics-final.json"),
        )
        .with_callback("restore_metrics", RestoreMetricsCallback(metrics_callback="metrics"))
    )


def _build_launch(
    cli: CliContext, *, work_dir: str = MixedMidtrainingRecipeConfig.work_dir
) -> BeakerLaunchConfig | None:
    if cli.cluster == "local":
        return None
    preset = get_preset("olmo-ddp")
    launch = build_launch_config(
        name=cli.run_name,
        cmd=cli.remote_cmd,
        cluster=cli.cluster,
        root_dir="/weka/oe-training-default",
        workspace="ai2/molmofication",
        num_nodes=2,
        step_timeout=None,
        step_soft_timeout=None,
    )
    if preset.beaker_image is not None:
        launch.beaker_image = preset.beaker_image
    launch.post_setup = preset.post_setup
    env = {item.name: item.value for item in launch.env_vars}
    env.update(dict(preset.env_vars))
    env.update(
        {
            "OLMO_CORE_DATA_VERIFICATION_CACHE_DIR": str(Path(work_dir) / "data-verification"),
            "OLMO_USE_OWN_SYMM_MEM": "1",
            "NVSHMEM_REMOTE_TRANSPORT": "none",
            "OLMO_EP_MP_HIGH_PRIORITY_GROUP": "1",
            "OLMO_OWN_SYMM_PREWARM": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "TORCH_LOGS": "-dynamo",
        }
    )
    launch.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
    launch.priority = "urgent"
    launch.min_runtime = "8h"
    launch.shared_memory = "32GiB"
    launch.env_secrets = [
        secret
        for secret in launch.env_secrets
        if secret.name not in {"BEAKER_TOKEN", "WANDB_API_KEY"}
    ] + [
        BeakerEnvSecret(name="BEAKER_TOKEN", secret="JASONR_BEAKER_TOKEN", required=True),
        BeakerEnvSecret(name="WANDB_API_KEY", secret="RUSTINS_WANDB_API_KEY", required=True),
    ]
    launch.aws_config_secret = None
    launch.aws_credentials_secret = None
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
    if loader.text_only != (recipe.text_loss_share == 1.0):
        raise OLMoConfigurationError("data_loader.text_only must match recipe.text_loss_share")
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
        raise OLMoConfigurationError("Use recipe.parent_checkpoint to select the initial model")
    if (
        config.trainer.load_strategy != LoadStrategy.always
        or config.trainer.load_optim_state is not False
        or config.trainer.load_trainer_state is not False
    ):
        raise OLMoConfigurationError("The alignment handoff requires model-only checkpoint loading")
    if config.pretraining_checkpoint != ancestry:
        raise OLMoConfigurationError("Pretraining ancestry must match the alignment parent")
    assert recipe.parent_checkpoint is not None
    paths = []
    for value in (recipe.parent_checkpoint, config.trainer.save_folder):
        path = normalize_path(value).rstrip("/")
        paths.append(path if is_url(path) else str(Path(path).resolve()))
    source_path, output_path = paths
    if (
        source_path == output_path
        or source_path.startswith(output_path + "/")
        or output_path.startswith(source_path + "/")
    ):
        raise OLMoConfigurationError("Use a separate output folder for mixed midtraining")


def build_config(cli: CliContext) -> MixedMidtrainingExperimentConfig:
    """Build mixed midtraining from checkpoint metadata and ordinary component overrides.

    No token arrays or visual datasets are opened. Dataset preparation and checkpoint
    loading remain the responsibility of the standard experiment runner.
    """
    overrides = _clean_opts(cli.overrides)
    recipe = _build_recipe(cli, overrides)
    parent, ancestry = _resolve_parent(recipe)
    tokenizer = _parent_tokenizer(parent, ancestry)
    revision = _tokenizer_revision(recipe, parent)
    loader = _build_data_loader(cli, recipe)
    if (
        type(loader.global_batch_size) is not int
        or loader.global_batch_size <= 0
        or loader.global_batch_size % recipe.sequence_length
    ):
        raise OLMoConfigurationError(
            "Global batch size must contain a positive whole number of sequences"
        )
    budget = (
        (recipe.max_tokens + loader.global_batch_size - 1) // loader.global_batch_size
    ) * loader.global_batch_size
    text = _build_text_dataset(recipe, tokenizer, budget, loader.global_batch_size)
    model = _build_model(parent)
    means = dict(DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS) if recipe.text_loss_share < 1.0 else {}
    if text.label_mask_paths is None:
        means[TEXT_SOURCE_NAME] = float(recipe.sequence_length - 1)
    elif recipe.text_loss_share == 1.0:
        means[TEXT_SOURCE_NAME] = 1.0
    replaced_sources = any(name in {"dataset", "dataset.sources"} for name, _ in overrides)
    visual_sources = (
        build_visual_sources(
            sequence_length=recipe.sequence_length,
            max_crops=recipe.max_crops,
            alignment_artifact_root=recipe.alignment_artifact_root,
            midtraining_artifact_root=recipe.midtraining_artifact_root,
        )
        if recipe.text_loss_share < 1.0 and not replaced_sources
        else {}
    )
    dataset = MultimodalMixtureConfig(
        tokenizer=text.tokenizer.copy(),
        sources=dict(
            sorted(
                {
                    **visual_sources,
                    TEXT_SOURCE_NAME: PretrainingReplayConfig(dataset=text.copy(), split="all"),
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
        launch=_build_launch(cli, work_dir=recipe.work_dir),
        model=model,
        dataset=dataset,
        data_loader=loader,
        train_module=_build_train_module(
            recipe.sequence_length,
            budget,
            loader.global_batch_size,
            text_only=recipe.text_loss_share == 1.0,
        ),
        trainer=_build_trainer(cli, recipe, budget),
        recipe=recipe,
        pretraining_checkpoint=ancestry,
        init_seed=6198,
    ).merge(cli.overrides)
    reusable_visual_calibration = (
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
