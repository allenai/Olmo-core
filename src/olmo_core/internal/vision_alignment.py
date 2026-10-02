"""Bridge, perception, and joint vision alignment with the internal experiment runner."""

import copy
import json
import logging
from dataclasses import dataclass, field, fields, replace
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url, normalize_path, resource_path
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
)
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.kda import KimiDeltaAttentionConfig
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import Molmo2TokenIds, MultimodalLMConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, PerGroupScheduler
from olmo_core.optim.multimodal_optimizer import MultimodalOLMoDDPOptimizerConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
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
    MultimodalEvaluatorCallbackConfig,
    MultimodalMetricSaverCallback,
    MultimodalWandBCallback,
    RestoreMetricsCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
)

from .common import build_launch_config
from .experiment import CliContext, ExperimentConfig
from .vision_alignment_data import (
    ALIGNMENT_LOSS_TARGETS,
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    build_visual_sources,
)

log = logging.getLogger(__name__)

_DOLMA2_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"

MULTIMODAL_OVERRIDES: dict[str, str] = {
    # Resolved-config keys (fnmatch patterns over the flattened ``ExperimentConfig`` dict, with
    # the text LM at ``model.lm``) where an alignment phase built from ``recipe.text_config``
    # differs from the text mid-training config it inherits from. Everything else is inherited
    # verbatim; ``vision_alignment_olmo35_test.py`` asserts this table is exact.
    "_CLASS_": "alignment experiment config class",
    "run_name": "the alignment run's own name",
    "recipe": "alignment recipe inputs",
    "pretraining_checkpoint": "recorded pretraining ancestry",
    "init_seed": "seeds the vision encoder / connector bootstrap",
    "launch": "built by the launcher (image, resources and env inherited; see _build_launch)",
    "model._CLASS_": "multimodal wrapper around the text LM",
    "model.vision": "vision encoder",
    "model.connector": "vision-to-language connector",
    "model.image_patch_token_id": "image patch token id from the tokenizer",
    "model.vit_layers": "vision features taken from these ViT layers",
    "model.lm.block*.routed_experts_router.lb_loss_weight": (
        "router load balancing off while the LM is frozen (bridge, perception); joint "
        "restores the pretrained coefficients"
    ),
    "model.lm.block*.sequence_mixer.use_experimental_kernels": (
        "document mode passes cu_seqlens to KDA; kernel_fun.kda.chunk_kda's is_supported() "
        "evaluates the cu_seqlens tensor as a bool and crashes, so the FLA kernels are used "
        "until kernel_fun handles packed documents"
    ),
    "train_module._CLASS_": "multimodal OLMoDDP train module",
    "train_module.rank_microbatch_size": "4 x seq (joint 2 x seq): vision tower cost per sequence",
    "train_module.trim_microbatch_image_padding": "multimodal microbatch image padding trim",
    "train_module.freeze_params": "phase policy: what trains",
    "train_module.train_embedding_rows": "image token rows are the only trainable embeddings",
    "train_module.vision_activation_checkpointing": "vision tower memory",
    "train_module.connector_activation_checkpointing": "off: wrapper breaks reset_parameters under the OLMoDDP init order",
    "train_module.response_logits_only": "logits only on supervised positions",
    "train_module.diagnostics_interval": "per-step multimodal diagnostics",
    "train_module.loss_group_weights": "joint text/vision loss split",
    "train_module.source_loss_mass_targets": "per-source loss mass targets",
    "train_module.scheduler": "per-group cosine schedules (connector / vision / LM)",
    "train_module.optim._CLASS_": "per-group clipping and partial master sync",
    "train_module.optim.lr": "phase learning rate",
    "train_module.optim.group_overrides": "connector / vision / embedding groups and rates",
    "train_module.optim.foreach_chunk_size": "optimizer memory with the vision tower resident",
    "train_module.optim.clip_grad_norm_by_scheduler_group": (
        "connector and vision gradients are on different scales"
    ),
    "trainer.save_folder": "alignment output folder",
    "trainer.work_dir": "alignment dataset cache",
    "trainer.load_path": "phase parent (bridge starts from the pretraining checkpoint)",
    "trainer.load_strategy": "phase handoff policy",
    "trainer.load_optim_state": "phase handoff policy",
    "trainer.load_trainer_state": "phase handoff policy",
    "trainer.max_duration": "phase step budget",
    "trainer.callbacks.checkpointer._CLASS_": "alignment checkpointer subclass (phase retention)",
    "trainer.callbacks.beaker._CLASS_": "alignment Beaker subclass (W&B config on resumed runs)",
    "trainer.callbacks.checkpointer.save_interval": "short phases: 500",
    "trainer.callbacks.checkpointer.ephemeral_save_interval": "short phases: 50",
    "trainer.callbacks.checkpointer.max_checkpoints": "phase retention",
    "trainer.callbacks.wandb": "alignment W&B project, auto-resume",
    "trainer.callbacks.slack_notifier": "text run's notifier is not carried",
    "trainer.callbacks.metrics": "multimodal metric saver",
    "trainer.callbacks.restore_metrics": "resume metrics",
    "trainer.callbacks.multimodal_evaluator": "in-loop multimodal evaluation",
    "trainer.callbacks.initialize_multimodal": "bridge bootstrap of vision and connector",
    "dataset": "multimodal mixture (visual sources, native text replay in joint)",
    "data_loader": "multimodal mixture loader (packing, crops); workers inherited",
}
"""Where an alignment config built from ``recipe.text_config`` differs from the text config."""


class AlignmentPhase(StrEnum):
    """Successive training phases before mixed vision/text midtraining."""

    bridge = "bridge"
    perception = "perception"
    joint = "joint"


@dataclass
class VisionAlignmentRecipeConfig(Config):
    """Inputs used to construct an alignment experiment's ordinary component configs.

    Set ``pretraining_checkpoint`` for bridge and ``parent_checkpoint`` for subsequent
    phases. The latter inherit the multimodal model and original text-data ancestry.
    Component-level CLI overrides are applied after these defaults are constructed.
    """

    phase: AlignmentPhase = AlignmentPhase.bridge
    text_config: str | None = None
    """Path to the text team's resolved mid-training ``config.json`` (a saved
    :class:`~olmo_core.internal.experiment.ExperimentConfig`).

    When set, the language model config and every text-side training setting (optimizer,
    train module, trainer bookkeeping, loader workers, launch image and resources) are
    inherited from it; the alignment phase changes only what :data:`MULTIMODAL_OVERRIDES`
    lists. The LM must match ``pretraining_checkpoint``, whose weights are loaded. Without it
    the LM config comes from the checkpoint's own ``config.json`` (legacy keys normalized,
    EMO routing cleared as text mid-training does) and the recipe's legacy defaults apply.
    """
    sequence_length: int | None = None
    """Context length for sources, packing, training, and evaluation.

    ``None`` uses 8,192 tokens for every phase. The global batch is 128 sequences.
    A context override preserves the phase's microbatch instance count and requires fresh
    visual loss-weight calibration; it does not modify RoPE.
    """
    pretraining_checkpoint: str | None = None
    parent_checkpoint: str | None = None
    artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT
    output_root: str = (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints"
    )
    work_dir: str = "/weka/oe-training-default/rustin/dataset-cache/vision-alignment"
    hf_cache_dir: str | None = "/weka/oe-training-default/rustin/hf-cache/hub"
    tokenizer_revision: str | None = None
    vision_model_id: str = "google/siglip2-so400m-patch14-384"
    vision_revision: str = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    text_validation_size: int = 1024
    """Native replay windows withheld in joint; zero requires separate text validation."""
    text_validation_seed: int = 6198
    """Seed shared by the complementary native replay training and validation splits."""
    router_lb_loss_weight: float | None = None
    """Override routed experts' load-balancing coefficients.

    ``None`` uses the phase policy: off for bridge, inherited for perception, and restored
    from pretraining for joint unless ``restore_pretraining_router_lb=False``.

    Zero disables the load-balancing objective without changing router z-loss or CE.
    This model-config override persists in checkpoints, including across phase handoffs.
    """
    restore_pretraining_router_lb: bool | None = None
    """Restore pretrained per-layer load-balancing coefficients after model inheritance.

    ``None`` restores in joint unless a coefficient is explicitly overridden. ``False``
    preserves the parent's coefficients. Restoration changes only ``lb_loss_weight``, not
    dispatch capacity or other objectives. Explicit restoration and coefficient overrides
    are mutually exclusive.
    """


@dataclass
class VisionAlignmentExperimentConfig(ExperimentConfig):
    """An alignment experiment with portable pretraining-checkpoint ancestry."""

    model: MultimodalLMConfig  # type: ignore[assignment]
    dataset: MultimodalMixtureConfig  # type: ignore[assignment]
    data_loader: MixtureDataLoaderConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    recipe: VisionAlignmentRecipeConfig = field(default_factory=VisionAlignmentRecipeConfig)
    pretraining_checkpoint: str = ""

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], overrides: Optional[List[str]] = None
    ) -> "VisionAlignmentExperimentConfig":
        """
        Build the config from a saved dictionary. A local run has no launch config, and
        ``as_config_dict()`` omits ``None`` fields, so ``launch`` is restored as ``None``.
        """
        return super().from_dict({"launch": None, **data}, overrides=overrides)


@dataclass(frozen=True)
class _PhaseDefaults:
    microbatch_instances: int
    steps: int
    freeze_params: tuple[str, ...]
    connector_lr: float
    vision_lr: float
    lm_lr: float
    connector_warmup: int
    vision_warmup: int
    lm_warmup: int
    sequence_length: int = 8192
    pack_max_crops: int = 64
    connector_decay_steps: int | None = None
    validation_sequence_length: int | None = None


_PHASES = {
    AlignmentPhase.bridge: _PhaseDefaults(
        microbatch_instances=4,
        steps=500,
        freeze_params=("vision.*", "lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"),
        connector_lr=2e-4,
        vision_lr=0.0,
        lm_lr=0.0,
        connector_warmup=100,
        vision_warmup=100,
        lm_warmup=100,
        connector_decay_steps=250,
        validation_sequence_length=2560,
    ),
    AlignmentPhase.perception: _PhaseDefaults(
        microbatch_instances=4,
        steps=2000,
        freeze_params=("lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"),
        connector_lr=5e-5,
        vision_lr=3e-6,
        lm_lr=0.0,
        connector_warmup=100,
        vision_warmup=250,
        lm_warmup=250,
        validation_sequence_length=2560,
    ),
    AlignmentPhase.joint: _PhaseDefaults(
        microbatch_instances=2,
        steps=1500,
        freeze_params=("lm.lm_head.w_out.weight",),
        connector_lr=2e-5,
        vision_lr=2e-6,
        lm_lr=1e-6,
        connector_warmup=50,
        vision_warmup=100,
        lm_warmup=100,
    ),
}


def _read_checkpoint_config(checkpoint: str) -> dict:
    with resource_path(checkpoint, "config.json").open() as stream:
        return json.load(stream)


def _load_text_config(path: str) -> dict:
    """Read a text mid-training config: a saved experiment config, or a dump wrapping one."""
    folder, _, fname = normalize_path(path).rpartition("/")
    with resource_path(folder or ".", fname).open() as stream:
        data = json.load(stream)
    if "model" not in data and isinstance(data.get("config"), dict):
        data = data["config"]
    for section in ("model", "train_module", "trainer"):
        if section not in data:
            raise OLMoConfigurationError(f"recipe.text_config lacks the {section!r} section")
    return data


def _checkpoint_lm_config(checkpoint: str) -> dict:
    """The checkpoint's LM config as text mid-training uses it: legacy keys normalized and
    EMO routing cleared."""
    model = copy.deepcopy(_read_checkpoint_config(checkpoint)["model"])
    _normalize_legacy_latent_moe_config(model)  # in place
    for block in [model.get("block"), *(model.get("block_overrides") or {}).values()]:
        router = (block or {}).get("routed_experts_router")
        if isinstance(router, dict) and "emo" in router:
            router["emo"] = None
    return model


def _check_text_lm_matches_checkpoint(lm: OLMoDDPModelConfig, checkpoint: str) -> None:
    pretrained = OLMoDDPModelConfig.from_dict(_checkpoint_lm_config(checkpoint))
    for name in ("d_model", "n_layers", "vocab_size", "tie_word_embeddings"):
        if getattr(lm, name) != getattr(pretrained, name):
            raise OLMoConfigurationError(
                f"recipe.text_config LM {name}={getattr(lm, name)!r} does not match the "
                f"pretraining checkpoint ({getattr(pretrained, name)!r})"
            )


def _image_token_rows(ids: Molmo2TokenIds) -> list[int]:
    return [
        ids.im_start_id,
        ids.im_end_id,
        ids.im_patch_id,
        ids.im_col_id,
        ids.low_res_im_start_id,
        ids.image_placeholder_id,
    ]


def _resolve_parent(recipe: VisionAlignmentRecipeConfig) -> tuple[str, dict | None]:
    if recipe.phase == AlignmentPhase.bridge:
        if recipe.parent_checkpoint is not None or not recipe.pretraining_checkpoint:
            raise OLMoConfigurationError(
                "Bridge requires recipe.pretraining_checkpoint and no recipe.parent_checkpoint"
            )
        return recipe.pretraining_checkpoint, None
    if not recipe.parent_checkpoint:
        raise OLMoConfigurationError("Set recipe.parent_checkpoint to the previous alignment phase")
    parent = _read_checkpoint_config(recipe.parent_checkpoint)
    previous = parent.get("recipe", {}).get("phase", parent.get("phase"))
    expected = "bridge" if recipe.phase == AlignmentPhase.perception else "perception"
    if previous != expected:
        raise OLMoConfigurationError(
            f"{recipe.phase} requires a {expected} checkpoint, got {previous}"
        )
    checkpoint = parent.get("pretraining_checkpoint") or parent.get("artifacts", {}).get(
        "base_checkpoint"
    )
    if not checkpoint:
        raise OLMoConfigurationError("Parent does not record its original pretraining checkpoint")
    if recipe.pretraining_checkpoint not in (None, checkpoint):
        raise OLMoConfigurationError("Requested pretraining checkpoint differs from phase parent")
    return checkpoint, parent


def _resolve_tokenizer_revision(
    recipe: VisionAlignmentRecipeConfig, parent: dict | None, tokenizer: TokenizerConfig
) -> str | None:
    if parent is not None:
        parent_dataset = parent.get("dataset", {})
        parent_artifacts = parent.get("artifacts", {})
        parent_tokenizer = parent_dataset.get("tokenizer")
        if parent_tokenizer is not None:
            if TokenizerConfig.from_dict(parent_tokenizer) != tokenizer:
                raise OLMoConfigurationError(
                    "Phase parent tokenizer differs from its pretraining checkpoint"
                )
        elif parent_artifacts.get("tokenizer_id", tokenizer.identifier) != tokenizer.identifier:
            raise OLMoConfigurationError(
                "Phase parent tokenizer differs from its pretraining checkpoint"
            )
        if "tokenizer_revision" in parent_dataset or "tokenizer_revision" in parent_artifacts:
            revision = parent_dataset.get(
                "tokenizer_revision", parent_artifacts.get("tokenizer_revision")
            )
            if recipe.tokenizer_revision is not None and recipe.tokenizer_revision != revision:
                raise OLMoConfigurationError(
                    "Requested tokenizer revision differs from the phase parent"
                )
            return revision
    return recipe.tokenizer_revision or (
        _DOLMA2_REVISION if tokenizer.identifier == "allenai/dolma2-tokenizer" else None
    )


def _build_model(
    recipe: VisionAlignmentRecipeConfig,
    checkpoint: str,
    parent: dict | None,
    token_ids: Molmo2TokenIds,
    text: dict | None = None,
) -> MultimodalLMConfig:
    if parent is not None:
        model = MultimodalLMConfig.from_dict(parent["model"])
    else:
        lm_dict = text["model"] if text is not None else _checkpoint_lm_config(checkpoint)
        lm = OLMoDDPModelConfig.from_dict(lm_dict)
        if not isinstance(lm, OLMoDDPModelConfig):
            raise OLMoConfigurationError("This alignment recipe currently requires an OLMoDDP LM")
        if text is not None:
            _check_text_lm_matches_checkpoint(lm, checkpoint)
        model = MultimodalLMConfig.molmo2_vision_stack(
            lm, image_patch_token_id=token_ids.im_patch_id
        )
    if not isinstance(model.lm, OLMoDDPModelConfig):
        raise OLMoConfigurationError("This alignment recipe currently requires an OLMoDDP LM")
    lb_loss_weight = recipe.router_lb_loss_weight
    if recipe.phase == AlignmentPhase.bridge and lb_loss_weight is None:
        lb_loss_weight = 0.0
    blocks: list[OLMoDDPTransformerBlockConfig] = []
    for block in [model.lm.block, *(model.lm.block_overrides or {}).values()]:
        if not isinstance(block, OLMoDDPTransformerBlockConfig):
            raise OLMoConfigurationError("Alignment requires OLMoDDP transformer blocks")
        blocks.append(block)
    document_mode = any(isinstance(b.sequence_mixer, KimiDeltaAttentionConfig) for b in blocks)
    for block in blocks:
        if block.routed_experts_router is not None and lb_loss_weight is not None:
            block.routed_experts_router.lb_loss_weight = lb_loss_weight
        if document_mode and isinstance(block.sequence_mixer, KimiDeltaAttentionConfig):
            # Packed examples reach KDA as documents (``cu_seqlens``). ``kernel_fun.kda.chunk_kda``
            # (``use_experimental_kernels=True``, the text team's setting) does not support
            # packed documents and its ``is_supported()`` check evaluates the ``cu_seqlens``
            # tensor as a bool ("Boolean value of Tensor with more than one value is
            # ambiguous"), so the FLA kernels are used until kernel_fun handles cu_seqlens.
            block.sequence_mixer.use_experimental_kernels = False
    # The multimodal wrapper feeds embeddings into the LM, which two-batch overlap cannot take.
    model.lm.two_batch_overlap = False
    return model


def _build_train_module(
    phase: AlignmentPhase,
    token_ids: Molmo2TokenIds,
    sequence_length: int,
    text: dict | None = None,
) -> MultimodalOLMoDDPTrainModuleConfig:
    policy = _PHASES[phase]
    # Text-side settings: inherited from the text config, else the recipe's legacy defaults.
    if text is not None:
        text_module = text["train_module"]
        text_optim = text_module["optim"]
        optim_settings: dict[str, Any] = {
            name: text_optim[name]
            for name in (
                "betas",
                "eps",
                "weight_decay",
                "compile",
                "sigma_factor",
                "max_grad_norm",
                "use_distributed",
                "check_nan_inf_grad",
                "rolling_interval_length",
                "reset_optimizer_moments_on_load",
            )
            if name in text_optim
        }
        if "betas" in optim_settings:
            optim_settings["betas"] = tuple(optim_settings["betas"])
        if "dtype" in text_optim:
            optim_settings["dtype"] = DType(text_optim["dtype"])
        module_settings: dict[str, Any] = {
            name: text_module[name]
            for name in (
                "z_loss_multiplier",
                "compile_model",
                "max_grad_norm",
                "reset_optimizer_states_on_load",
                "reset_optimizer_states_on_resume",
                "label_ignore_index",
            )
            if name in text_module
        }
        module_settings["dp_config"] = (
            TransformerDataParallelConfig.from_dict(text_module["dp_config"])
            if text_module.get("dp_config") is not None
            else None
        )
        module_settings["ep_config"] = (
            TransformerExpertParallelConfig.from_dict(text_module["ep_config"])
            if text_module.get("ep_config") is not None
            else None
        )
    else:
        optim_settings = dict(
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            compile=False,
            sigma_factor=12,
            max_grad_norm=1.0,
            check_nan_inf_grad=True,
            use_distributed=True,
        )
        module_settings = dict(
            z_loss_multiplier=1e-4,
            compile_model=True,
            max_grad_norm=1.0,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp,
                reduce_dtype=DType.float32,
                only_allreduce_last_microbatch=True,
                reduce_grads_in_fp32=True,
                accumulate_grads_in_fp32=True,
            ),
            ep_config=TransformerExpertParallelConfig(degree=8),
        )
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=policy.microbatch_instances * sequence_length,
        max_sequence_length=sequence_length,
        optim=MultimodalOLMoDDPOptimizerConfig(
            lr=policy.lm_lr or policy.connector_lr,
            group_overrides=[
                OptimGroupOverride(
                    params=["*lm.embeddings.weight"],
                    opts={
                        "lr": policy.connector_lr,
                        "weight_decay": 0.0,
                        "scheduler_name": "connector",
                    },
                ),
                OptimGroupOverride(
                    params=["*connector.*"],
                    opts={
                        "lr": policy.connector_lr,
                        "weight_decay": 0.0,
                        "scheduler_name": "connector",
                    },
                ),
                OptimGroupOverride(
                    params=["*vision.*"],
                    opts={"lr": policy.vision_lr, "weight_decay": 0.0, "scheduler_name": "vision"},
                ),
            ],
            foreach_chunk_size=50_000_000,
            clip_grad_norm_by_scheduler_group=True,
            **optim_settings,
        ),
        freeze_params=list(policy.freeze_params),
        train_embedding_rows=_image_token_rows(token_ids),
        vision_activation_checkpointing=phase != AlignmentPhase.bridge,
        # Off on purpose: ``VisionConnector.apply_activation_checkpointing`` wraps the pooling and
        # projector modules in ``checkpoint_wrapper``, and ``reset_parameters`` then no longer
        # recognizes the projector (``isinstance`` on the wrapper fails), so a connector initialized
        # after wrapping -- the OLMoDDP train module's order -- stays uninitialized (connector output
        # RMS 0, step-1 grad norm 5e3 in the bridge check). The connector's activations are a
        # negligible share of memory here, so checkpointing them buys nothing.
        connector_activation_checkpointing=False,
        response_logits_only=True,
        diagnostics_interval=1,
        loss_group_weights={"text": 0.35, "vision": 0.65}
        if phase == AlignmentPhase.joint
        else None,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(
                    warmup=policy.connector_warmup,
                    alpha_f=0.1,
                    t_max=policy.connector_decay_steps or policy.steps,
                ),
                "vision": CosWithWarmup(
                    warmup=policy.vision_warmup, alpha_f=0.1, t_max=policy.steps
                ),
            },
            default=CosWithWarmup(warmup=policy.lm_warmup, alpha_f=0.1, t_max=policy.steps),
        ),
        **module_settings,
    )


def _restore_pretraining_router_lb(
    model: OLMoDDPModelConfig, pretrained: OLMoDDPModelConfig
) -> None:
    """Restore effective per-layer balancing coefficients without changing the LM function."""
    model_fields = (
        "name",
        "d_model",
        "n_layers",
        "vocab_size",
        "embedding_norm",
        "embed_scale",
        "tie_word_embeddings",
    )
    if any(getattr(model, name) != getattr(pretrained, name) for name in model_fields):
        raise OLMoConfigurationError(
            "Pretraining router LB restoration requires matching LM topology"
        )

    def topology(block: OLMoDDPTransformerBlockConfig):
        value = block.as_config_dict()
        # Alignment changes execution choices, not learned architecture. Do not restore any
        # of these fields, and permit larger fixed capacities or checkpointing policies.
        for name in (
            "ep",
            "rowwise_fp8",
            "checkpoint_attn",
            "checkpoint_permute_moe_unpermute",
            "checkpoint_second_unpermute",
        ):
            value.pop(name, None)
        if isinstance(block.sequence_mixer, AttentionConfig):
            value["sequence_mixer"].pop("backend", None)
        elif isinstance(block.sequence_mixer, KimiDeltaAttentionConfig):
            value["sequence_mixer"].pop("use_experimental_kernels", None)
        for name in ("routed_experts_router", "shared_experts_router"):
            if (router := value.get(name)) is not None:
                for coefficient in ("lb_loss_weight", "z_loss_weight", "orth_loss_weight"):
                    router.pop(coefficient, None)
        return value

    current_blocks = model.resolved_block_configs
    pretrained_blocks = pretrained.resolved_block_configs
    for index, (current, original) in enumerate(zip(current_blocks, pretrained_blocks)):
        if not isinstance(current, OLMoDDPTransformerBlockConfig) or not isinstance(
            original, OLMoDDPTransformerBlockConfig
        ):
            raise OLMoConfigurationError("Router LB restoration requires OLMoDDP block topology")
        if topology(current) != topology(original):
            raise OLMoConfigurationError(
                f"Pretraining router LB restoration requires matching block/router topology "
                f"at layer {index}"
            )
        router = original.routed_experts_router
        if (
            router is not None
            and router.lb_loss_weight is not None
            and (not isfinite(router.lb_loss_weight) or router.lb_loss_weight < 0)
        ):
            raise OLMoConfigurationError(
                f"Invalid pretrained router LB coefficient at layer {index}"
            )

    # A default block can be shared by several effective layers. Preserve that layout unless
    # differing original coefficients require a new layer override; never overwrite an earlier
    # layer through the same shared config object.
    assigned: dict[int, float | None] = {}
    for index, (current, original) in enumerate(zip(current_blocks, pretrained_blocks)):
        assert isinstance(current, OLMoDDPTransformerBlockConfig)
        assert isinstance(original, OLMoDDPTransformerBlockConfig)
        if current.routed_experts_router is None:
            continue
        assert original.routed_experts_router is not None
        coefficient = original.routed_experts_router.lb_loss_weight
        if id(current) in assigned and assigned[id(current)] != coefficient:
            current = current.copy()
            model.block_overrides = dict(model.block_overrides or {})
            model.block_overrides[index] = current
        assigned[id(current)] = coefficient
        assert current.routed_experts_router is not None
        current.routed_experts_router = current.routed_experts_router.copy()
        current.routed_experts_router.lb_loss_weight = coefficient


def _build_recipe(cli: CliContext) -> VisionAlignmentRecipeConfig:
    recipe = VisionAlignmentRecipeConfig().merge(
        [arg.replace("--recipe.", "--", 1) for arg in cli.overrides if arg.startswith("--recipe.")]
    )
    # Reject conflicting policy before loading model or dataset metadata.
    if recipe.restore_pretraining_router_lb and (
        recipe.router_lb_loss_weight is not None
        or any(
            arg.startswith("--model.") and arg.partition("=")[0].endswith(".lb_loss_weight")
            for arg in cli.overrides
        )
    ):
        raise OLMoConfigurationError(
            "restore_pretraining_router_lb is mutually exclusive with explicit router LB overrides"
        )
    if recipe.router_lb_loss_weight is not None and (
        not isfinite(recipe.router_lb_loss_weight) or recipe.router_lb_loss_weight < 0
    ):
        raise OLMoConfigurationError("recipe.router_lb_loss_weight must be finite and nonnegative")
    return recipe


def _sequence_length(recipe: VisionAlignmentRecipeConfig) -> int:
    policy = _PHASES[recipe.phase]
    sequence_length = (
        policy.sequence_length if recipe.sequence_length is None else recipe.sequence_length
    )
    if type(sequence_length) is not int or sequence_length < 2:
        raise OLMoConfigurationError("recipe.sequence_length must be an integer of at least two")
    return sequence_length


def _build_datasets(
    recipe: VisionAlignmentRecipeConfig,
    checkpoint: str,
    parent: dict | None,
    sequence_length: int,
) -> tuple[MultimodalMixtureConfig, MultimodalMixtureConfig, Molmo2TokenIds]:
    phase = recipe.phase
    policy = _PHASES[phase]
    replay = PretrainingReplayConfig(
        checkpoint=checkpoint, sequence_length=sequence_length, work_dir=recipe.work_dir
    )
    text = replay.resolve_dataset()
    lm_config = _read_checkpoint_config(checkpoint)["model"]
    dataset = MultimodalMixtureConfig(
        tokenizer=text.tokenizer,
        tokenizer_revision=_resolve_tokenizer_revision(recipe, parent, text.tokenizer),
        tokenizer_cache_dir=recipe.hf_cache_dir,
        model_vocab_size=lm_config["vocab_size"],
        sources=build_visual_sources(phase, sequence_length, recipe.artifact_root),
        target_loss_mass=ALIGNMENT_LOSS_TARGETS[phase].copy(),
        mean_loss_weight=ALIGNMENT_MEAN_LOSS_WEIGHTS[phase].copy(),
    )
    if phase == AlignmentPhase.joint:
        if recipe.text_validation_size < 0:
            raise OLMoConfigurationError("recipe.text_validation_size cannot be negative")
        if recipe.text_validation_size:
            replay.split = "train"
            replay.validation_size = recipe.text_validation_size
            replay.split_seed = recipe.text_validation_seed
        dataset.sources["native_text_replay"] = replay
        dataset.sources = dict(sorted(dataset.sources.items()))
    if (
        text.tokenizer != TokenizerConfig.dolma2()
        or dataset.tokenizer_revision != _DOLMA2_REVISION
        or recipe.artifact_root != DEFAULT_ALIGNMENT_ARTIFACT_ROOT
        or sequence_length != policy.sequence_length
    ):
        dataset.mean_loss_weight = {}
    if phase == AlignmentPhase.joint:
        if text.label_mask_paths is None:
            dataset.mean_loss_weight["native_text_replay"] = float(sequence_length - 1)
        else:
            dataset.mean_loss_weight.pop("native_text_replay", None)
    _, token_ids = dataset.build_tokenizer()
    validation = dataset.copy()
    validation_sequence_length = (
        sequence_length
        if policy.validation_sequence_length is None
        else min(sequence_length, policy.validation_sequence_length)
    )
    validation.sources = build_visual_sources(
        phase, validation_sequence_length, recipe.artifact_root, split="validation"
    )
    validation.target_loss_mass = {name: 1.0 for name in validation.sources}
    validation.mean_loss_weight = {}
    return dataset, validation, token_ids


def _build_data_loader(
    cli: CliContext,
    recipe: VisionAlignmentRecipeConfig,
    sequence_length: int,
    text: dict | None = None,
) -> MixtureDataLoaderConfig:
    workers = 8
    if text is not None and "num_workers" in text.get("data_loader", {}):
        workers = int(text["data_loader"]["num_workers"])
    return MixtureDataLoaderConfig(
        global_batch_size=128 * sequence_length,
        sequence_length=sequence_length,
        seed=95818,
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        pack=True,
        pack_buffer_size=48,
        pack_max_crops=_PHASES[recipe.phase].pack_max_crops,
        # Molmo2's Stage-1 packing objective, its continuous source stream (exact resume
        # from the checkpointed cursor) and the collator metadata the train module reads.
        pack_image_weight=1.0,
        continuous_stream=True,
        batch_metadata=True,
        prefetch_workers=workers,
        max_consecutive_data_errors=0,
        max_total_data_errors=0,
        group_sequence_quotas={"text": 16, "vision": 112}
        if recipe.phase == AlignmentPhase.joint
        else None,
    )


def _build_trainer(
    cli: CliContext,
    recipe: VisionAlignmentRecipeConfig,
    checkpoint: str,
    validation: MultimodalMixtureConfig,
    token_ids: Molmo2TokenIds,
    sequence_length: int,
    text: dict | None = None,
) -> TrainerConfig:
    phase = recipe.phase
    policy = _PHASES[phase]
    is_bridge = phase == AlignmentPhase.bridge
    is_joint = phase == AlignmentPhase.joint
    # Bookkeeping settings and callbacks come from the text config; the text run's own
    # checkpointer cadence, W&B, notifier and Beaker callback are replaced below.
    bookkeeping: dict[str, Any] = dict(metrics_collect_interval=1, cancel_check_interval=5)
    inherited_callbacks: dict[str, Any] = {
        "gpu_monitor": GPUMemoryMonitorCallback(),
        "config_saver": ConfigSaverCallback(),
        "garbage_collector": GarbageCollectorCallback(),
    }
    checkpointer = MultimodalCheckpointerCallback(
        save_interval=500,
        ephemeral_save_interval=50,
        save_async=False,
        pre_train_checkpoint=False,
        max_checkpoints=3 if is_joint else 2,
    )
    if text is not None:
        text_trainer = TrainerConfig.from_dict(text["trainer"])
        bookkeeping = dict(
            checkpointer=text_trainer.checkpointer,
            metrics_collect_interval=text_trainer.metrics_collect_interval,
            cancel_check_interval=text_trainer.cancel_check_interval,
            async_bookkeeping=text_trainer.async_bookkeeping,
            bookkeeping_soft_timeout=text_trainer.bookkeeping_soft_timeout,
            save_overwrite=text_trainer.save_overwrite,
        )
        inherited_callbacks = {
            name: callback
            for name, callback in text_trainer.callbacks.items()
            if name not in ("checkpointer", "wandb", "slack_notifier", "beaker")
        }
        text_checkpointer = text_trainer.callbacks.get("checkpointer")
        if isinstance(text_checkpointer, CheckpointerCallback):
            # The text run's checkpointer settings, in the alignment subclass (retention).
            checkpointer = MultimodalCheckpointerCallback(
                **{
                    f.name: getattr(text_checkpointer, f.name)
                    for f in fields(CheckpointerCallback)
                    if f.init and not f.name.startswith("_")
                }
            )
            checkpointer = replace(
                checkpointer,
                save_interval=500,
                ephemeral_save_interval=50,
                max_checkpoints=3 if is_joint else 2,
            )
    trainer = TrainerConfig(
        save_folder=f"{recipe.output_root}/{cli.run_name}",
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        load_path=recipe.parent_checkpoint,
        load_strategy=LoadStrategy.if_available if is_bridge else LoadStrategy.always,
        load_optim_state=None if is_bridge else False,
        load_trainer_state=None if is_bridge else False,
        max_duration=Duration.steps(policy.steps),
        **bookkeeping,
    )
    for name, callback in inherited_callbacks.items():
        trainer = trainer.with_callback(name, callback)
    trainer = (
        trainer.with_callback("checkpointer", checkpointer)
        .with_callback(
            "metrics",
            MultimodalMetricSaverCallback(
                save_interval=1, final_metrics_fname="metrics-final.json"
            ),
        )
        .with_callback("restore_metrics", RestoreMetricsCallback(metrics_callback="metrics"))
        .with_callback("beaker", MultimodalBeakerCallback())
        .with_callback(
            "wandb",
            MultimodalWandBCallback(
                name=cli.run_name, project="vision-alignment", auto_resume=True
            ),
        )
        .with_callback(
            "multimodal_evaluator",
            MultimodalEvaluatorCallbackConfig(
                eval_dataset=validation,
                sequence_length=sequence_length,
                rank_batch_size=1 if is_joint else policy.microbatch_instances,
                examples_per_source=64,
                eval_interval=500,
                eval_on_startup=True,
                eval_on_finish=True,
                blank_image_sources=["pixmo_caption", "pixmo_transcript"],
                matched_image_sources=["pixmo_caption", "pixmo_transcript"],
            ),
        )
    )
    if is_bridge:
        trainer = trainer.with_callback(
            "initialize_multimodal",
            InitializeMultimodalModelCallback(
                language_checkpoint=checkpoint,
                vision_model_id=recipe.vision_model_id,
                vision_revision=recipe.vision_revision,
                cache_dir=recipe.hf_cache_dir,
                image_token_ids=_image_token_rows(token_ids),
                seed=6198,
            ),
        )
    return trainer


_ALIGNMENT_WORKSPACE = "ai2/oe-olmo3p5-mt"
_ALIGNMENT_BUDGET = "ai2/oe-other"
_ALIGNMENT_SECRETS = [
    BeakerEnvSecret(name="BEAKER_TOKEN", secret="jasonr_BEAKER_TOKEN", required=True),
    BeakerEnvSecret(name="WANDB_API_KEY", secret="jasonr_WANDB_API_KEY", required=True),
]


def _build_launch(
    cli: CliContext,
    *,
    work_dir: str = VisionAlignmentRecipeConfig.work_dir,
    text: dict | None = None,
) -> BeakerLaunchConfig | None:
    if cli.cluster == "local":
        return None
    text_launch = None
    if text is not None and text.get("launch") is not None:
        text_launch = BeakerLaunchConfig.from_dict(text["launch"])
    preset = get_preset("olmo-ddp")
    launch_kwargs: dict[str, Any] = {}
    if text_launch is not None:
        launch_kwargs["beaker_image"] = text_launch.beaker_image
    elif preset.beaker_image is not None:
        launch_kwargs["beaker_image"] = preset.beaker_image
    launch = build_launch_config(
        name=cli.run_name,
        cmd=cli.remote_cmd,
        cluster=cli.cluster,
        root_dir="/weka/oe-training-default",
        workspace=_ALIGNMENT_WORKSPACE,
        budget=_ALIGNMENT_BUDGET,
        num_nodes=text_launch.num_nodes if text_launch is not None else 2,
        step_timeout=None,
        step_soft_timeout=None,
        **launch_kwargs,
    )
    env: dict[str, str] = {}
    if text_launch is not None:
        # Image, install step, resources and environment come from the text run; the source
        # layout differs (the alignment scripts import from the cloned tree), so PYTHONPATH is
        # the recipe's own.
        launch.beaker_image = text_launch.beaker_image
        launch.post_setup = text_launch.post_setup
        launch.num_gpus = text_launch.num_gpus
        launch.shared_memory = text_launch.shared_memory
        launch.priority = text_launch.priority
        launch.google_credentials_secret = text_launch.google_credentials_secret
        for bucket in text_launch.weka_buckets:
            if all(existing.bucket != bucket.bucket for existing in launch.weka_buckets):
                launch.weka_buckets.append(
                    BeakerWekaBucket(bucket=bucket.bucket, mount=bucket.mount)
                )
        env.update(
            (entry.name, entry.value)
            for entry in text_launch.env_vars
            if entry.name != "PYTHONPATH"
        )
    else:
        if preset.beaker_image is not None:
            launch.beaker_image = preset.beaker_image
        launch.post_setup = preset.post_setup
        launch.priority = "urgent"
        launch.shared_memory = "32GiB"
        env.update(preset.env_vars)
        env.update(
            {
                "OLMO_USE_OWN_SYMM_MEM": "1",
                "OLMO_EP_MP_HIGH_PRIORITY_GROUP": "1",
                "OLMO_OWN_SYMM_PREWARM": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "8",
                "TORCH_LOGS": "-dynamo",
            }
        )
    env.update(
        {
            # The alignment scripts import the recipe modules from the cloned source tree.
            "PYTHONPATH": "/gantry-runtime/src",
            "OLMO_CORE_DATA_VERIFICATION_CACHE_DIR": str(Path(work_dir) / "data-verification"),
        }
    )
    launch.env_vars = [entry for entry in launch.env_vars if entry.name not in env] + [
        BeakerEnvVar(name=k, value=v) for k, v in env.items()
    ]
    launch.min_runtime = "8h"
    launch.follow = False
    launch.env_secrets = [
        secret
        for secret in launch.env_secrets
        if secret.name not in {"BEAKER_TOKEN", "WANDB_API_KEY"}
    ] + [secret.copy() for secret in _ALIGNMENT_SECRETS]
    launch.aws_config_secret = None
    launch.aws_credentials_secret = None
    return launch


def build_config(cli: CliContext) -> VisionAlignmentExperimentConfig:
    """Build one alignment phase from checkpoint metadata and ordinary CLI overrides."""
    recipe = _build_recipe(cli)
    sequence_length = _sequence_length(recipe)
    checkpoint, parent = _resolve_parent(recipe)
    text = _load_text_config(recipe.text_config) if recipe.text_config else None
    dataset, validation, token_ids = _build_datasets(recipe, checkpoint, parent, sequence_length)
    config = VisionAlignmentExperimentConfig(
        run_name=cli.run_name,
        launch=_build_launch(cli, work_dir=recipe.work_dir, text=text),
        model=_build_model(recipe, checkpoint, parent, token_ids, text=text),
        dataset=dataset,
        data_loader=_build_data_loader(cli, recipe, sequence_length, text=text),
        train_module=_build_train_module(recipe.phase, token_ids, sequence_length, text=text),
        trainer=_build_trainer(
            cli, recipe, checkpoint, validation, token_ids, sequence_length, text=text
        ),
        recipe=recipe,
        pretraining_checkpoint=checkpoint,
        init_seed=6198,
        **{
            # Process-level settings of the text run, where this tree's experiment config has them.
            name: text[name]
            for name in ("backend", "process_group_timeout_seconds")
            if text is not None
            and name in text
            and name in {f.name for f in fields(VisionAlignmentExperimentConfig)}
        },
    ).merge(cli.overrides)
    _validate_config(config, cli, dataset, token_ids, checkpoint)
    return config


def _validate_config(
    config: VisionAlignmentExperimentConfig,
    cli: CliContext,
    dataset: MultimodalMixtureConfig,
    token_ids: Molmo2TokenIds,
    checkpoint: str,
) -> None:
    recipe = config.recipe
    phase = recipe.phase
    policy = _PHASES[phase]
    sequence_length = _sequence_length(recipe)
    explicit_router_lb = recipe.router_lb_loss_weight is not None or any(
        arg.startswith("--model.") and arg.partition("=")[0].endswith(".lb_loss_weight")
        for arg in cli.overrides
    )
    if recipe.restore_pretraining_router_lb and explicit_router_lb:
        raise OLMoConfigurationError(
            "restore_pretraining_router_lb is mutually exclusive with explicit router LB overrides"
        )
    if recipe.restore_pretraining_router_lb is None:
        recipe.restore_pretraining_router_lb = (
            phase == AlignmentPhase.joint and not explicit_router_lb
        )
    if recipe.restore_pretraining_router_lb:
        if not isinstance(config.model.lm, OLMoDDPModelConfig):
            raise OLMoConfigurationError("Router LB restoration requires an OLMoDDP LM")
        pretrained = OLMoDDPModelConfig.from_dict(_checkpoint_lm_config(checkpoint))
        _restore_pretraining_router_lb(config.model.lm, pretrained)
    if (
        config.dataset.tokenizer != dataset.tokenizer
        or config.dataset.tokenizer_revision != dataset.tokenizer_revision
    ):
        raise OLMoConfigurationError(
            "Select the tokenizer through recipe.pretraining_checkpoint and recipe.tokenizer_revision"
        )
    changed_sources = [
        name
        for name, source in config.dataset.sources.items()
        if source != dataset.sources.get(name)
        and not any(
            arg.startswith((f"--dataset.mean_loss_weight.{name}=", "--dataset.mean_loss_weight="))
            for arg in cli.overrides
        )
    ]
    if changed_sources:
        raise OLMoConfigurationError(
            f"Supply calibrated dataset.mean_loss_weight for changed sources: {changed_sources}"
        )
    for name, source in config.dataset.sources.items():
        if isinstance(source, PretrainingReplayConfig):
            replay_dataset = source.resolve_dataset()
            if replay_dataset.sequence_length != config.data_loader.sequence_length:
                raise OLMoConfigurationError("Replay and data-loader sequence lengths must agree")
            if replay_dataset.tokenizer != config.dataset.tokenizer:
                raise OLMoConfigurationError("Replay and visual tokenizers must agree")
            if replay_dataset.label_mask_paths is None:
                expected_mean = float(replay_dataset.sequence_length - 1)
                if config.dataset.mean_loss_weight.get(name, expected_mean) != expected_mean:
                    raise OLMoConfigurationError(
                        f"Unmasked replay source {name!r} has exact mean_loss_weight "
                        f"{expected_mean} (sequence_length - 1)"
                    )
                config.dataset.mean_loss_weight[name] = expected_mean
    if sequence_length != policy.sequence_length:
        missing = sorted(set(config.dataset.sources) - set(config.dataset.mean_loss_weight))
        if missing:
            raise OLMoConfigurationError(
                "Changing recipe.sequence_length requires calibrated dataset.mean_loss_weight "
                f"for sources: {missing}. Use MultimodalMixtureConfig.estimate_mean_loss_weights() "
                "for a bounded calibration sample."
            )
    config.dataset.sampling_weights()
    targets = config.dataset.target_loss_mass
    config.train_module.source_loss_mass_targets = {
        k: v / sum(targets.values()) for k, v in targets.items()
    }
    if phase == AlignmentPhase.joint and not any(
        arg.startswith("--data_loader.source_groups") for arg in cli.overrides
    ):
        config.data_loader.source_groups = {
            name: "text" if isinstance(source, PretrainingReplayConfig) else "vision"
            for name, source in config.dataset.sources.items()
        }
    if config.trainer.load_path != recipe.parent_checkpoint:
        raise OLMoConfigurationError(
            "Use recipe.parent_checkpoint to select a phase's initial model"
        )
    source_path = normalize_path(recipe.parent_checkpoint or checkpoint).rstrip("/")
    output_path = normalize_path(config.trainer.save_folder).rstrip("/")
    if not is_url(source_path):
        source_path = str(Path(source_path).resolve())
    if not is_url(output_path):
        output_path = str(Path(output_path).resolve())
    if (
        source_path == output_path
        or source_path.startswith(output_path + "/")
        or output_path.startswith(source_path + "/")
    ):
        raise OLMoConfigurationError("Use a separate output folder for the new alignment phase")
    if config.pretraining_checkpoint != checkpoint:
        raise OLMoConfigurationError("Pretraining ancestry must match the selected phase parent")
    if config.model.lm.tie_word_embeddings:
        raise OLMoConfigurationError(
            "Alignment's row-masked image embeddings require untied LM input and output weights"
        )
    if config.model.image_patch_token_id != token_ids.im_patch_id or (
        config.train_module.train_embedding_rows != _image_token_rows(token_ids)
    ):
        raise OLMoConfigurationError(
            "Model image tokens and trainable rows must match the tokenizer"
        )
    if config.dataset.model_vocab_size != config.model.lm.vocab_size:
        raise OLMoConfigurationError("Dataset and model vocabulary sizes must agree")
    if config.data_loader.sequence_length != config.train_module.max_sequence_length:
        raise OLMoConfigurationError("Data-loader and train-module sequence lengths must agree")
    evaluator = config.trainer.callbacks["multimodal_evaluator"]
    if (
        evaluator.eval_dataset.tokenizer != config.dataset.tokenizer
        or evaluator.eval_dataset.tokenizer_revision != config.dataset.tokenizer_revision
        or evaluator.eval_dataset.model_vocab_size != config.model.lm.vocab_size
    ):
        raise OLMoConfigurationError("Evaluation tokenizer and vocabulary must match training")
    if evaluator.sequence_length != config.data_loader.sequence_length:
        raise OLMoConfigurationError("Evaluation and training sequence lengths must agree")
    if phase == AlignmentPhase.joint and recipe.text_validation_size:
        training_replay = config.dataset.sources.get("native_text_replay")
        if (
            not isinstance(training_replay, PretrainingReplayConfig)
            or training_replay.split != "train"
        ):
            raise OLMoConfigurationError("Native replay validation requires the training split")
        if training_replay.validation_size < evaluator.examples_per_source:
            raise OLMoConfigurationError("Native replay holdout must cover examples_per_source")
        # Derive after CLI overrides so corpus, window boundaries, and exclusion seed agree.
        holdout = training_replay.copy()
        holdout.split = "validation"
        evaluator.eval_dataset.sources["native_text_holdout"] = holdout
        evaluator.eval_dataset.target_loss_mass["native_text_holdout"] = 1.0
