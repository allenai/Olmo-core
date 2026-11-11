"""Olmo 3.5 multimodal SFT: the text team's current SFT recipe plus stage-2 image data.

The model, routers (EMo included) and the train module's numerics come from the base
checkpoint's own saved ``config.json`` (a mixed mid-training checkpoint), so the SFT trains
exactly the network that checkpoint holds. Only the data, optimizer settings, schedule and
trainer change:

* **Text.** open-instruct's frozen C12 numpy cache plus the 10% long-context replay dir, as the
  text team's GPU SFT trains them (``olmo_core_finetune.py`` at open-instruct ``h182-gpu``):
  half a C12 pass (260,848 C12 instances), the replay riding on top in proportion, each row one
  document (attention and EMo), ``<think>``/``</think>`` as single promoted tokens whose rows are
  seeded from their old pieces before the first step.
* **Images.** Named sources with calibrated mean loss weight and mean tokens per example (the
  data track's stage-2 mixture; a stand-in until it lands). The image dose in tokens is the knob;
  the text share of supervised-loss mass and the total steps follow from it.
* **Optimizer.** The checkpoint's optimizer with the text SFT's settings: peak LM LR 2e-4,
  weight decay 0, grad clipping 1.0, linear warmup over 3% of the steps then linear to 0;
  connector LR 2x and vision LR 1/10 of the LM's, on the same schedule.

``text_order="open_instruct"`` trains text only, in open-instruct's own data order (same cache,
seed and global batch give the same instances at every step), for the step-matching gate and for
a text-only control in this trainer.

Usage (under torchrun for ``train``; ``dry_run`` prints the config)::

    python src/scripts/train/Olmo35-Multimodal-SFT.py train RUN_NAME ai2/holmes \\
        --recipe.base=64k [--recipe.image_dose_tokens=3.5e9] [OVERRIDES...]
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import numpy as np

from olmo_core.config import Config, DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig
from olmo_core.data.multimodal.sft_numpy_cache import SftNumpyCacheConfig
from olmo_core.data.multimodal.sft_text_loader import SftTextDataLoaderConfig
from olmo_core.distributed.utils import barrier, get_local_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import resource_path
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, PerGroupScheduler
from olmo_core.optim.multimodal_optimizer import MultimodalOLMoDDPOptimizerConfig
from olmo_core.train import (
    CheckpointerConfig,
    Duration,
    LoadStrategy,
    Trainer,
    TrainerConfig,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.multimodal import (
    MultimodalCheckpointerCallback,
    MultimodalMetricSaverCallback,
)
from olmo_core.train.callbacks.promoted_rows import OLMO35_THINK_ROWS, PromotedTokenRowsCallback
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
)
from olmo_core.utils import seed_all

from .experiment import CliContext, ExperimentConfig, SubCmd
from .vision_alignment import parse_cli_args

log = logging.getLogger(__name__)

__all__ = [
    "Olmo35MultimodalSFTRecipeConfig",
    "Olmo35MultimodalSFTExperimentConfig",
    "ImageSourceConfig",
    "derive_schedule",
    "build_config",
    "train",
    "main",
]

BASES: Dict[str, str] = {
    "64k": "/weka/olmo-3p5-mid-checkpoints/jasonr/olmo35-multimodal-midtraining/checkpoints/"
    "olmoe3-810m-cx2-mm-merged-mt-lc-v6-lm1e4/step20000",
    "256k-emo-causal": "/weka/olmo-3p5-mid-checkpoints/jasonr/olmo35-multimodal-midtraining/"
    "checkpoints/olmoe3-810m-cx2-mm-hero-mt-lc-256k-emocausal-lm1e4/step22240",
}
"""The two multimodal mid-training checkpoints SFT starts from (native OLMo-core)."""

C12_CACHE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/a7/cache-neptune/numpy_sft/"
    "bf82a8fab8-6068a350-rowaligned"
)
"""The frozen C12 numpy cache (open-instruct resolves ``bf82a8fab8-6068a350-rowaligned`` in its
deletable cache to this directory)."""
C12_INSTANCES = 449_487
C12_LOSS_TOKENS = 26_787_776_256
C12_HALF_PASS = 260_848
"""Half a C12 pass: 16,303 steps x 16 instances, the text-only GPU runs' C12 exposure."""
REPLAY_DIR = "/weka/olmo-3p5-post-checkpoints/abhishekr/h182/replay-b1c12r"
REPLAY_INSTANCES = 58_480
REPLAY_LOSS_TOKENS = 2_678_796_581
"""The long-context replay dir of the text-only GPU runs: one olmOCR science document plus EOS
per row, every position trained; 0.10 of C12's expected consumed loss tokens."""
TOKENIZER = TokenizerConfig(
    vocab_size=100278,
    eos_token_id=100257,
    pad_token_id=100277,
    identifier="allenai/dolma2-tokenizer-olmo35",
)
TOKENIZER_REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
TEXT_SOURCE = "text_sft"
TEXT_INSTANCES = 294_560
"""Text instances (C12 + replay) the text-only GPU run's seed 34521 consumed: 18,410 steps x 16
(its offline loader replay, S_r)."""
TEXT_MEAN_LOSS_WEIGHT = 17_090_511_368 / 294_560
"""Measured mean loss weight per text instance (58,021): that replay's consumed loss tokens,
15,546,575,242 C12 plus 1,543,936,126 replay, over its 294,560 instances."""
RESERVED_SLOT_TOKENS = ["<think>", "</think>"]


@dataclass
class ImageSourceConfig(Config):
    """One image source of the SFT mixture and its calibration."""

    source: Config
    """A mixture source config (``build(tokenizer)``)."""
    loss_share: float
    """Relative share of the image loss mass (normalized over image sources)."""
    mean_loss_weight: Optional[float] = None
    """Mean ``sum(loss_masks)`` per example; measured at build when unset."""
    mean_tokens: Optional[float] = None
    """Mean tokens per example; measured at build when unset."""


@dataclass
class Olmo35MultimodalSFTRecipeConfig(Config):
    base: str = "64k"
    """A key of :data:`BASES`, or a checkpoint directory."""
    text_cache: str = C12_CACHE
    replay_dirs: List[str] = field(default_factory=lambda: [REPLAY_DIR])
    text_instances: Optional[int] = TEXT_INSTANCES
    """Text instances (C12 + replay) to draw. ``None``: the half pass scaled to include the replay
    in proportion, ``round(C12_HALF_PASS * (C12 + replay) / C12)``."""
    text_mean_loss_weight: Optional[float] = TEXT_MEAN_LOSS_WEIGHT
    """Mean loss weight per text instance. ``None``: from the cache totals."""
    image_data: str = "stage2_v1"
    """``stage2_v1`` (:mod:`~olmo_core.data.multimodal.mixtures.olmo35_sft_mm_v1`) or
    ``stand_in`` (stage-1 sources from the base checkpoint's config; mechanics only)."""
    text_order: str = "mixture"
    """``mixture`` (text and images drawn by the mixture loader) or ``open_instruct`` (text only,
    in the text SFT's order)."""
    data_seed: int = 34521
    image_dose_tokens: float = 3.5e9
    """Image tokens (packed positions holding image examples) over the run."""
    image_pack_fill: float = 0.95
    """Expected fraction of an image sequence's positions that hold tokens (the rest pad)."""
    image_sources: Dict[str, ImageSourceConfig] = field(default_factory=dict)
    """Empty: the stand-in (:func:`stand_in_image_sources`)."""
    calibration_samples: int = 32
    lm_lr: float = 2e-4
    warmup_ratio: float = 0.03
    connector_lr_scale: float = 2.0
    vision_lr_divisor: float = 10.0
    max_grad_norm: float = 1.0
    schedule_steps: Optional[int] = None
    """LR horizon and run length. Default: derived from the text exposure and the image dose."""
    stop_at_step: Optional[int] = None
    """Hard stop for smokes and gates (the schedule keeps its full horizon)."""
    sequence_length: int = 65_536
    num_nodes: int = 2
    gpus_per_node: int = 8
    compile_model: bool = False
    """open-instruct's text SFT runs uncompiled (``--no_compile_model``)."""
    save_interval: int = 1000
    ephemeral_save_interval: Optional[int] = None
    output_root: str = "/weka/oe-training-default/jasonr/olmo35-mm-sft/checkpoints"
    work_dir: str = "/weka/oe-training-default/jasonr/olmo35-mm-sft/dataset-cache"
    hf_cache_dir: str = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/hub"
    promoted_rows: bool = True
    batch_hash_dir: Optional[str] = None
    """Write each step's per-rank batch hashes (and instance indices) here (gates)."""
    step1_reference: bool = False
    """Also dump step 1's batch for an HF reference and score it in eval mode (gate a)."""
    prefetch_workers: int = 8


@dataclass
class Olmo35MultimodalSFTExperimentConfig(ExperimentConfig):
    model: MultimodalLMConfig  # type: ignore[assignment]
    dataset: Optional[MultimodalMixtureConfig]  # type: ignore[assignment]
    data_loader: Any  # MixtureDataLoaderConfig | SftTextDataLoaderConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig  # type: ignore[assignment]
    recipe: Olmo35MultimodalSFTRecipeConfig = field(
        default_factory=Olmo35MultimodalSFTRecipeConfig
    )
    schedule: Dict[str, Any] = field(default_factory=dict)
    """The derived numbers (text share, steps, image examples), for the record."""


def base_checkpoint(base: str) -> str:
    return BASES.get(base, base)


def read_checkpoint_config(checkpoint: str) -> dict[str, Any]:
    with resource_path(checkpoint, "config.json").open() as stream:
        return json.load(stream)


def text_totals(recipe: Olmo35MultimodalSFTRecipeConfig) -> tuple[int, int, float]:
    """``(instances in the text dataset, instances drawn, mean loss weight per instance)``."""
    replay = recipe.replay_dirs == [REPLAY_DIR]
    if recipe.replay_dirs and not replay and recipe.text_mean_loss_weight is None:
        raise OLMoConfigurationError("Other replay dirs need recipe.text_mean_loss_weight")
    total = C12_INSTANCES + (REPLAY_INSTANCES if replay else 0)
    loss = C12_LOSS_TOKENS + (REPLAY_LOSS_TOKENS if replay else 0)
    drawn = recipe.text_instances or round(C12_HALF_PASS * total / C12_INSTANCES)
    mean = recipe.text_mean_loss_weight or loss / total
    return total, drawn, mean


def derive_schedule(
    *,
    text_instances: int,
    text_mean_loss_weight: float,
    image_dose_tokens: float,
    image_stats: Dict[str, tuple[float, float, float]],
    sequence_length: int,
    instances_per_step: int,
    image_pack_fill: float = 0.95,
) -> Dict[str, Any]:
    """Derive the loss-mass targets and the run length from the text exposure and image dose.

    :param image_stats: ``name -> (loss_share, mean_loss_weight, mean_tokens)``.
    :returns: ``text_loss_share``, ``target_loss_mass``, ``image_examples``,
        ``image_sequences``, ``steps`` and the expected loss masses.
    """
    if text_instances <= 0 or text_mean_loss_weight <= 0:
        raise ValueError("Text exposure must be positive")
    text_mass = text_instances * text_mean_loss_weight
    if not image_stats or image_dose_tokens <= 0:
        steps = math.ceil(text_instances / instances_per_step)
        return dict(
            text_loss_share=1.0,
            target_loss_mass={TEXT_SOURCE: 1.0},
            text_instances=text_instances,
            text_loss_mass=text_mass,
            image_examples=0,
            image_sequences=0,
            image_loss_mass=0.0,
            steps=steps,
        )
    shares = {name: s for name, (s, _, _) in image_stats.items()}
    total_share = sum(shares.values())
    # Example probabilities within the image sources: share / mean loss weight.
    raw = {name: shares[name] / total_share / image_stats[name][1] for name in shares}
    norm = sum(raw.values())
    prob = {name: value / norm for name, value in raw.items()}
    tokens_per_example = sum(prob[n] * image_stats[n][2] for n in prob)
    loss_per_example = sum(prob[n] * image_stats[n][1] for n in prob)
    image_examples = image_dose_tokens / tokens_per_example
    image_mass = image_examples * loss_per_example
    share = text_mass / (text_mass + image_mass)
    image_sequences = image_dose_tokens / (sequence_length * image_pack_fill)
    steps = math.ceil((text_instances + image_sequences) / instances_per_step)
    targets = {TEXT_SOURCE: share}
    targets.update({n: (1 - share) * shares[n] / total_share for n in shares})
    return dict(
        text_loss_share=share,
        target_loss_mass=targets,
        text_instances=text_instances,
        text_loss_mass=text_mass,
        image_examples=image_examples,
        image_sequences=image_sequences,
        image_loss_mass=image_mass,
        image_tokens_per_example=tokens_per_example,
        image_loss_per_example=loss_per_example,
        steps=steps,
    )


STAND_IN_SOURCES = ("pixmo_points_v2", "cosyn_chart_exp", "text_rich_doc", "pixmo_count_v2")
"""Stand-in image sources until the stage-2 chat-format mixture lands: stage-1 sources from the
base checkpoint's own saved config (exactly configured for this model and tokenizer), equal
loss shares. Mechanics only; they are not the SFT image data."""


def stand_in_image_sources(raw: dict[str, Any], sequence_length: int) -> Dict[str, ImageSourceConfig]:
    """Stage-1 sources from the base checkpoint's saved config, at ``sequence_length``."""
    saved = raw["dataset"]["sources"]
    means = raw["dataset"].get("mean_loss_weight", {})
    out: Dict[str, ImageSourceConfig] = {}
    for name in STAND_IN_SOURCES:
        if name not in saved:
            continue
        source = MultimodalMixtureConfig.from_dict(
            {"tokenizer": raw["dataset"]["tokenizer"], "sources": {name: saved[name]}}
        ).sources[name]

        def set_length(component: Config):
            if hasattr(component, "max_sequence_length"):
                component.max_sequence_length = sequence_length  # type: ignore[attr-defined]

        source.apply(set_length)
        out[name] = ImageSourceConfig(source=source, loss_share=1.0, mean_loss_weight=means.get(name))
    if not out:
        raise OLMoConfigurationError("None of the stand-in sources is in the checkpoint config")
    return out


def stage2_v1_image_sources(dose_tokens: float, sequence_length: int) -> Dict[str, ImageSourceConfig]:
    """The stage-2 v1 mixture: each source's loss share is its expected examples at the dose
    times its calibrated mean loss weight, so the mixture draws the expected examples."""
    from olmo_core.data.multimodal.mixtures import olmo35_sft_mm_v1 as v1

    if sequence_length != v1.SEQUENCE_LENGTH:
        raise OLMoConfigurationError(f"stage2_v1 is calibrated at {v1.SEQUENCE_LENGTH} tokens")
    examples = v1.expected_examples(dose_tokens)
    return {
        name: ImageSourceConfig(
            source=config,
            loss_share=examples[name] * v1.MEAN_LOSS_WEIGHTS[name],
            mean_loss_weight=v1.MEAN_LOSS_WEIGHTS[name],
            mean_tokens=v1.MEAN_TOKENS[name],
        )
        for name, config in v1.build_sources(max_sequence_length=sequence_length).items()
    }


def _measure_image_stats(
    dataset: MultimodalMixtureConfig, names: List[str], samples: int, seed: int = 0
) -> Dict[str, tuple[float, float]]:
    """Mean loss weight and mean tokens per example of ``names``, from a fixed sample."""
    tokenizer, token_ids = dataset.build_tokenizer()
    probe = MultimodalMixtureConfig(
        tokenizer=dataset.tokenizer,
        sources={name: dataset.sources[name] for name in names},
        tokenizer_revision=dataset.tokenizer_revision,
        tokenizer_cache_dir=dataset.tokenizer_cache_dir,
        model_vocab_size=dataset.model_vocab_size,
        # Placeholders: validate() wants every source in the targets and means.
        target_loss_mass={name: 1.0 / len(names) for name in names},
        mean_loss_weight={name: 1.0 for name in names},
    )
    stats = {}
    for name, built in probe.build_sources(tokenizer, token_ids).items():
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(built), min(samples, len(built)), replace=False)
        get = getattr(built, "get", None)
        weights, tokens = [], []
        for index in indices:
            example = get(int(index), 0) if callable(get) else built[int(index)]
            weights.append(float(np.asarray(example["loss_masks"], dtype=np.float64).sum()))
            tokens.append(float(len(example["input_ids"])))
        stats[name] = (float(np.mean(weights)), float(np.mean(tokens)))
    return stats


def _optimizer(raw_optim: dict, recipe: Olmo35MultimodalSFTRecipeConfig) -> MultimodalOLMoDDPOptimizerConfig:
    optim = MultimodalOLMoDDPOptimizerConfig.from_dict(raw_optim)
    optim.lr = recipe.lm_lr
    optim.weight_decay = 0.0
    optim.max_grad_norm = recipe.max_grad_norm
    overrides = []
    for override in optim.group_overrides or []:
        opts = dict(override.opts)
        if "*connector.*" in override.params:
            opts.update(lr=recipe.lm_lr * recipe.connector_lr_scale, weight_decay=0.0)
        elif "*vision.*" in override.params:
            opts.update(lr=recipe.lm_lr / recipe.vision_lr_divisor, weight_decay=0.0)
        elif opts.get("scheduler_name") == "embeddings":
            # Mid-training's separate embedding schedule: SFT trains them on the LM's.
            opts.pop("scheduler_name")
        if "weight_decay" in opts:
            opts["weight_decay"] = 0.0
        overrides.append(OptimGroupOverride(params=list(override.params), opts=opts))
    optim.group_overrides = overrides
    return optim


def build_config(cli: CliContext) -> Olmo35MultimodalSFTExperimentConfig:
    """Build the SFT experiment from the base checkpoint's saved config and the recipe."""
    recipe = Olmo35MultimodalSFTRecipeConfig().merge(list(cli.overrides), prefix="recipe")
    others = [o for o in cli.overrides if not o.lstrip("-").startswith("recipe.")]
    checkpoint = base_checkpoint(recipe.base)
    raw = read_checkpoint_config(checkpoint)
    L = recipe.sequence_length
    world = recipe.num_nodes * recipe.gpus_per_node
    if recipe.text_order not in ("mixture", "open_instruct"):
        raise OLMoConfigurationError(f"Unknown text_order {recipe.text_order!r}")

    model = MultimodalLMConfig.from_dict(raw["model"])
    _, text_drawn, text_mean = text_totals(recipe)
    cache = SftNumpyCacheConfig(
        numpy_dir=recipe.text_cache,
        work_dir=f"{recipe.work_dir}/{cli.run_name}/text",
        sequence_length=L,
        extra_numpy_dirs=list(recipe.replay_dirs),
        # Same ids as open-instruct's dolma2() (only pad/EOS matter for packing); one tokenizer
        # config in the saved experiment, as the HF converter requires.
        tokenizer=TOKENIZER,
    )
    run_dir = f"{recipe.output_root}/{cli.run_name}"

    dataset: Optional[MultimodalMixtureConfig] = None
    data_loader: Any
    image_sources: Dict[str, ImageSourceConfig] = {}
    if recipe.text_order == "open_instruct":
        schedule = derive_schedule(
            text_instances=text_drawn,
            text_mean_loss_weight=text_mean,
            image_dose_tokens=0,
            image_stats={},
            sequence_length=L,
            instances_per_step=world,
        )
        data_loader = SftTextDataLoaderConfig(
            cache=cache,
            global_batch_size=world * L,
            seed=recipe.data_seed,
            work_dir=f"{recipe.work_dir}/{cli.run_name}/loader",
            pad_token_id=TOKENIZER.pad_token_id,
        )
    else:
        if recipe.image_sources:
            image_sources = dict(recipe.image_sources)
        elif recipe.image_data == "stage2_v1":
            image_sources = stage2_v1_image_sources(recipe.image_dose_tokens, L)
        elif recipe.image_data == "stand_in":
            image_sources = stand_in_image_sources(raw, L)
        else:
            raise OLMoConfigurationError(f"Unknown image_data {recipe.image_data!r}")
        dataset = MultimodalMixtureConfig(
            tokenizer=TOKENIZER,
            sources={TEXT_SOURCE: cache, **{n: s.source for n, s in image_sources.items()}},
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_cache_dir=recipe.hf_cache_dir,
            model_vocab_size=model.lm.vocab_size,
            tokenizer_json_split=True,
            reserved_slot_tokens=list(RESERVED_SLOT_TOKENS),
        )
        missing = [
            n for n, s in image_sources.items() if s.mean_loss_weight is None or s.mean_tokens is None
        ]
        if missing:
            measured = _measure_image_stats(dataset, missing, recipe.calibration_samples)
            for name, (weight, tokens) in measured.items():
                source = image_sources[name]
                if source.mean_loss_weight is None:
                    source.mean_loss_weight = weight
                if source.mean_tokens is None:
                    source.mean_tokens = tokens
            log.info("Measured image source stats: %s", measured)
        schedule = derive_schedule(
            text_instances=text_drawn,
            text_mean_loss_weight=text_mean,
            image_dose_tokens=recipe.image_dose_tokens,
            image_stats={
                n: (s.loss_share, float(s.mean_loss_weight), float(s.mean_tokens))  # type: ignore[arg-type]
                for n, s in image_sources.items()
            },
            sequence_length=L,
            instances_per_step=world,
            image_pack_fill=recipe.image_pack_fill,
        )
        dataset.mean_loss_weight = {
            TEXT_SOURCE: text_mean,
            **{n: float(s.mean_loss_weight) for n, s in image_sources.items()},  # type: ignore[arg-type]
        }
        dataset.target_loss_mass = dict(schedule["target_loss_mass"])
        saved_loader = dict(raw["data_loader"])
        data_loader = MixtureDataLoaderConfig(
            global_batch_size=world * L,
            sequence_length=L,
            work_dir=f"{recipe.work_dir}/{cli.run_name}/loader",
            seed=recipe.data_seed,
            pack=True,
            pack_max_crops=saved_loader.get("pack_max_crops", 512) * L // int(saved_loader["sequence_length"]) or 512,
            pack_buffer_size=max(48, saved_loader.get("pack_buffer_size", 384) * L // int(saved_loader["sequence_length"])),
            pack_image_weight=saved_loader.get("pack_image_weight", 1.0),
            continuous_stream=True,
            batch_metadata=True,
            image_dtype=DType.bfloat16,
            compact_images=True,
            prefetch_workers=recipe.prefetch_workers,
            max_consecutive_data_errors=0,
            max_total_data_errors=0,
        )

    steps = recipe.schedule_steps or schedule["steps"]
    schedule["steps"] = steps
    warmup = int(steps * recipe.warmup_ratio)
    linear = LinearWithWarmup(warmup=warmup, alpha_f=0.0, t_max=steps)
    raw_tm = raw["train_module"]
    optim = _optimizer(raw_tm["optim"], recipe)
    scheduler_names = sorted(
        {o.opts["scheduler_name"] for o in optim.group_overrides or [] if "scheduler_name" in o.opts}
    )
    train_module = MultimodalOLMoDDPTrainModuleConfig.from_dict(
        {
            k: v
            for k, v in raw_tm.items()
            if k not in ("scheduler", "source_loss_mass_targets")
        }
    )
    train_module.optim = optim
    train_module.scheduler = PerGroupScheduler(
        schedulers={name: linear.copy() for name in scheduler_names},
        default=linear,
    )
    train_module.rank_microbatch_size = L
    train_module.max_sequence_length = L
    train_module.max_grad_norm = recipe.max_grad_norm
    train_module.compile_model = recipe.compile_model
    train_module.reset_optimizer_states_on_load = True
    # Router losses over loss tokens, as the text SFT trainer divides them (step-matching gate).
    train_module.router_loss_divisor = "loss_tokens"
    train_module.source_loss_mass_targets = (
        dict(schedule["target_loss_mass"]) if dataset is not None and len(schedule["target_loss_mass"]) > 1 else None
    )

    checkpointer = MultimodalCheckpointerCallback(
        save_interval=recipe.save_interval,
        ephemeral_save_interval=recipe.ephemeral_save_interval,
        save_async=False,
    )
    trainer = TrainerConfig(
        save_folder=run_dir,
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        save_overwrite=False,
        load_path=checkpoint,
        load_strategy=LoadStrategy.always,
        load_optim_state=False,
        load_trainer_state=False,
        max_duration=Duration.steps(steps),
        hard_stop=Duration.steps(recipe.stop_at_step) if recipe.stop_at_step else None,
        checkpointer=CheckpointerConfig(load_thread_count=8),
        metrics_collect_interval=1,
        cancel_check_interval=10,
    )
    trainer = (
        trainer.with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("checkpointer", checkpointer)
        .with_callback(
            "metrics",
            MultimodalMetricSaverCallback(save_interval=1, final_metrics_fname="metrics-final.json"),
        )
    )
    if recipe.promoted_rows:
        trainer = trainer.with_callback("promoted_rows", PromotedTokenRowsCallback(rows=list(OLMO35_THINK_ROWS)))
    if recipe.batch_hash_dir:
        from olmo_core.train.callbacks.batch_hash import BatchHashCallback

        trainer = trainer.with_callback("batch_hash", BatchHashCallback(
                out_dir=recipe.batch_hash_dir, step1_reference=recipe.step1_reference
            ))

    config = Olmo35MultimodalSFTExperimentConfig(
        run_name=cli.run_name,
        launch=None,
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
        trainer=trainer,
        recipe=recipe,
        schedule=schedule,
        init_seed=raw.get("init_seed", 12536),
    )
    return cast(Olmo35MultimodalSFTExperimentConfig, config.merge(others)) if others else config


def train(config: Olmo35MultimodalSFTExperimentConfig) -> Trainer:
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    if config.dataset is None:
        data_loader = config.data_loader.build(dp_process_group=train_module.dp_process_group)
    else:
        data_loader = config.data_loader.build(
            config.dataset.build(), dp_process_group=train_module.dp_process_group
        )
    barrier()
    trainer = config.trainer.build(train_module, data_loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    trainer.fit()
    return trainer


def main() -> None:
    cli = parse_cli_args()
    config = build_config(cli)
    cli.cmd.prepare_environment(config)
    if get_local_rank() == 0:
        print(config)
        print(json.dumps(config.schedule, indent=1, default=str))
    if cli.cmd == SubCmd.train:
        train(config)
        teardown_training_environment()
    elif cli.cmd != SubCmd.dry_run:
        raise OLMoConfigurationError(f"Unsupported command {cli.cmd}")
