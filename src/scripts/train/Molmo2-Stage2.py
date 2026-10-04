"""
Molmo2 "stage 2" SFT (reproduction of ``mm_olmo``'s ``image-only-v9`` mixture).

Fine-tunes connector + ViT + LLM on the image-only-v9 mixture with 16k sequence packing.
Defaults to a 3-dataset debug subset (``tulu4``, ``text_vqa``, ``chart_qa_weighted``) for
smoke tests; set ``--mixture=image-only-v9`` for the full 32-source mixture once parity
is green.

Quick local smoke test (2 GPUs, debug mixture, 5 steps)::

    torchrun --nproc-per-node=2 src/scripts/train/Molmo2-Stage2.py train smoke \\
        --trainer.max_duration.value=5 --trainer.max_duration.unit=steps \\
        --global_batch_size=32768 --train_module.rank_microbatch_size=16384 \\
        --train_module.compile_model=false

Resume weights from an OLMo-core stage-1 checkpoint (model only, fresh optimizer)::

    --trainer.load_path=/path/to/stage1/run

Set ``--trainer.load_path=null`` to initialise from HF ``allenai/Molmo2-4B`` instead.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Sequence, cast

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import MixtureDataLoader, MultimodalCollatorConfig
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    VALIDATION_MIXTURES,
    build_image_only_v9_mixture,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
)
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.nn.vision import MultimodalLM, MultimodalLMConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    PerGroupScheduler,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    UnshardedModelExportCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MultimodalTransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

MODEL_ID = "allenai/Molmo2-4B"
SEQUENCE_LENGTH = 16384  # mm_olmo image-only-v9: --seq_len 16384
USE_FLEX_ATTN = True
PACK_SEQUENCES = True
COMPILE_MODEL = True
RESPONSE_LOGITS_ONLY = True
DATA_PREFETCH_WORKERS = 4
MAX_CROPS = 8
# Per-pack crop capacity for the 2D-knapsack packer. mm_olmo derives this from the
# preprocessor's max per-example output; with multi-image enabled that is
# max_images * (1 global + max(MAX_CROPS, high_res_max_crops=24) local crops) =
# 5 * 25 = 125. In practice the 16k token budget binds first; the crop capacity
# mainly bounds worst-case collator/ViT memory. Setting it below a single example's
# crop count would force that example into its own mostly-padding 16k pack.
PACK_MAX_CROPS = 5 * (1 + 24)
EST_TOKENS_PER_EXAMPLE = 1500  # packed 16k sequences; tune if batch counts look off

# mm_olmo train_image_video_sft.py (image-only-v9): global 128, microbatch 2 per GPU.
GLOBAL_BATCH_INSTANCES = 128
RANK_MICROBATCH_INSTANCES = 2
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo SFT).
CONNECTOR_LR = 5e-6
VISION_LR = 5e-6
LLM_LR = 1e-5
COMPONENT_WARMUP = 200
ALPHA_F = 0.1

MAX_STEPS = 300_000

# Extra image-SFT sources beyond mm_olmo's image-only-v9 mixture, OFF by default.
# Rates are fractions of the total mixture: the 43 official sources are scaled by
# (1 - sum(extra rates)) and the extras are appended, so enabling them dilutes the
# official recipe proportionally. All read parquet shards straight from weka.
#
# MMFineReason-SFT: multimodal reasoning; supervision is the `<answer>` content of
# `original_answer` (the `<think>` trace is dropped) — see MMFineReasonDataset.
MMFINEREASON_RATE = 0.0
#
# FineVision configs -> sampling rate. Any config downloaded under FINEVISION_ROOT works;
# these five are verified. Rows are single-turn, one image each.
#   visualwebinstruct(filtered)  263,581  web visual instruction
#   mavis_math_rule_geo           99,986  synthetic geometry with CoT answers
#   mavis_math_metagen            87,348  synthetic math with CoT answers
#   geo170k(align)                35,297  geometry caption/alignment
#   geo170k(qa)                   12,101  geometry multiple-choice
# NOTE: ~13% of MMFineReason rows are re-annotations of visualwebinstruct(filtered) images,
# so enabling both double-samples those images (with different answers).
FINEVISION_RATES: dict = {
    "visualwebinstruct(filtered)": 0.0,
    "mavis_math_rule_geo": 0.0,
    "mavis_math_metagen": 0.0,
    "geo170k(align)": 0.0,
    "geo170k(qa)": 0.0,
}
# Optional quality floor applied to every enabled FineVision config (1-5 per-row minimums;
# None = keep all). `min_visual_dependency` is the most useful: it keeps answers that
# actually need the image. Do NOT use `min_image_correspondence` here — it is 1 for 75% of
# geo170k(qa) and 63% of mavis_math_metagen rows, which would discard most of them.
FINEVISION_MIN_VISUAL_DEPENDENCY: Optional[int] = None

# Default init: latest step under this OLMo-core stage-1 run (model weights only).
# HARDCODED personal checkpoint (donovanc's stage-1 run on weka). Point this at your own
# stage-1 run via --trainer.load_path=/path/to/run, or --trainer.load_path=null to
# initialise from the released HF Molmo2-4B weights instead.
DEFAULT_LOAD_PATH = (
    "/weka/oe-training-default/donovanc/molmofication/checkpoints/"
    "molmo2-pretraining-olmo-core/8-gpu-holmes/8-gpu-holmes-olmo-core-stable"
)

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 1
BEAKER_WORKSPACE = "ai2/OLMo-core"
BEAKER_BUDGET = "ai2/oe-other"

WANDB_PROJECT: Optional[str] = "molmo2-stage2"
WANDB_ENTITY: Optional[str] = None

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalTransformerTrainModuleConfig
    trainer: TrainerConfig
    model_id: str = MODEL_ID
    data_seed: int = 50189
    init_seed: int = 6198
    global_batch_size: int = GLOBAL_BATCH_SIZE
    mixture: str = "debug"
    """Mixture tier — see ``VALIDATION_MIXTURES`` in ``image_only_v9.py``."""
    pack_sequences: bool = PACK_SEQUENCES
    pack_max_crops: int = PACK_MAX_CROPS
    mmfinereason_rate: float = MMFINEREASON_RATE
    """Mixture fraction for MMFineReason-SFT (0 disables). The official image-only-v9
    sources are scaled by ``1 - (mmfinereason_rate + finevision_rate)``."""
    finevision_rate: float = 0.0
    """Total mixture fraction for the five verified FineVision configs, split evenly
    across them via ``FINEVISION_RATES`` keys (0 disables)."""


def _build_model_config() -> MultimodalLMConfig:
    from transformers import AutoConfig

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
    )

    ensure_default_rope_registered()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config = molmo2_config_from_hf_config(hf_config)
    # mm_olmo SFT fine-tuning setting: llm.residual_dropout = 0.1 (with
    # response_residual_dropout = 0.0, so a single residual-stream dropout matches).
    config.lm.block.dropout = 0.1
    return config


def _mixture_dataset_names(mixture: str) -> Optional[Sequence[str]]:
    if mixture not in VALIDATION_MIXTURES:
        known = ", ".join(sorted(VALIDATION_MIXTURES))
        raise ValueError(f"Unknown mixture {mixture!r}; use one of: {known}")
    return VALIDATION_MIXTURES[mixture]


def _full_image_only_v9_names(tokenizer, seed: int) -> List[str]:
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        IMAGE_ONLY_V9_SUBMIXTURES,
        build_image_only_v9_datasets,
        compute_flat_mixture_weights,
    )

    datasets_map = build_image_only_v9_datasets(
        tokenizer, seed, max_sequence_length=SEQUENCE_LENGTH
    )
    lengths = {name: len(datasets_map[name]) for name in datasets_map.keys()}
    flat = compute_flat_mixture_weights(IMAGE_ONLY_V9_SUBMIXTURES, lengths)
    return [name for name, _ in flat]


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    model_config = _build_model_config()

    collator_config = MultimodalCollatorConfig(
        pad_token_id=151643,
        label_ignore_index=-100,
        pad_sequence_length=SEQUENCE_LENGTH,
    )

    train_module_config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LLM_LR,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["connector.*"],
                    opts=dict(lr=CONNECTOR_LR, weight_decay=0.0, scheduler_name="connector"),
                ),
                OptimGroupOverride(
                    params=["vision.*"],
                    opts=dict(lr=VISION_LR, weight_decay=0.0, scheduler_name="vision"),
                ),
            ],
        ),
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=COMPILE_MODEL,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
                "vision": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
            },
            default=CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_blocks,
            block_interval=2,
        ),
        response_logits_only=RESPONSE_LOGITS_ONLY,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
            save_overwrite=True,
            load_path=DEFAULT_LOAD_PATH,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=2000, ephemeral_save_interval=500, save_async=False),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                enabled=WANDB_PROJECT is not None,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        # Every permanent checkpoint also gets `model.safetensors` + `olmo_core_config.json`, so
        # evals read one file instead of the 256-shard DCP state (olmo-eval issue #379).
        .with_callback("unsharded_export", UnshardedModelExportCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
    )

    launch_config = build_launch_config(
        name=run_name,
        root_dir=root_dir,
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=NUM_NODES,
    )
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets if s.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    launch_config.post_setup = "pip install -U 'datasets>=4,<6'"
    if USE_FLEX_ATTN:
        launch_config.env_vars = list(launch_config.env_vars) + [
            BeakerEnvVar(name="OLMO2_FLEX_ATTN", value="1")
        ]

    return ExperimentConfig(
        model=model_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
    ).merge(overrides)


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def _init_weights_from_hf(model: MultimodalLM, model_cfg: MultimodalLMConfig) -> None:
    from transformers import AutoModelForImageTextToText

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
        retie_word_embeddings,
    )

    ensure_default_rope_registered()
    log.info("Loading HF weights from %s ...", MODEL_ID)
    hf = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)
    reinit_rope_buffers(hf)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), model_cfg)
    del hf
    model.to_empty(device=get_default_device())
    model.load_state_dict(converted, strict=False)
    # `to_empty` silently un-ties tied word embeddings (Molmo2-4B); restore the share so
    # training updates the head and the embedding table as one parameter, like mm_olmo.
    retie_word_embeddings(model)
    del converted


def _build_mixture(tokenizer, config: ExperimentConfig):
    names_filter = _mixture_dataset_names(config.mixture)
    datasets, weights = build_image_only_v9_mixture(
        tokenizer,
        seed=config.data_seed,
        dataset_names=names_filter,
        max_sequence_length=SEQUENCE_LENGTH,
    )
    names = (
        _full_image_only_v9_names(tokenizer, config.data_seed)
        if names_filter is None
        else list(names_filter)
    )
    datasets, weights, names = _append_extra_sft_sources(
        config, tokenizer, datasets, weights, names
    )
    log.info(
        "Mixture %s sources / weights: %s",
        config.mixture,
        list(zip(names, [round(w, 4) for w in weights])),
    )
    return datasets, weights, names


def _append_extra_sft_sources(config: "ExperimentConfig", tokenizer, datasets, weights, names):
    """Append MMFineReason / FineVision at their configured rates (no-op when all 0).

    ``config.finevision_rate`` is split evenly across the configs named in
    ``FINEVISION_RATES``; a per-config rate in that dict adds on top (module-level
    fine-tuning knob for uneven splits).
    """
    from olmo_core.data.multimodal import (
        FineVisionDatasetConfig,
        MMFineReasonDatasetConfig,
    )

    per_config = config.finevision_rate / max(len(FINEVISION_RATES), 1)
    fv = {
        name: rate + per_config for name, rate in FINEVISION_RATES.items() if rate + per_config > 0
    }
    mmfr_rate = config.mmfinereason_rate
    extra_total = mmfr_rate + sum(fv.values())
    if extra_total <= 0:
        return datasets, weights, names
    if extra_total >= 1:
        raise ValueError(f"Extra SFT rates sum to {extra_total}; must be < 1")

    datasets = list(datasets)
    weights = [w * (1.0 - extra_total) for w in weights]
    names = list(names)
    if mmfr_rate > 0:
        datasets.append(
            MMFineReasonDatasetConfig(
                max_crops=MAX_CROPS, max_sequence_length=SEQUENCE_LENGTH
            ).build(tokenizer)
        )
        weights.append(mmfr_rate)
        names.append("mmfinereason")
    for cfg_name, rate in fv.items():
        datasets.append(
            FineVisionDatasetConfig(
                config_name=cfg_name,
                max_crops=MAX_CROPS,
                max_sequence_length=SEQUENCE_LENGTH,
                min_visual_dependency=FINEVISION_MIN_VISUAL_DEPENDENCY,
            ).build(tokenizer)
        )
        weights.append(rate)
        names.append(f"finevision[{cfg_name}]")
    return datasets, weights, names


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    tokenizer = _load_tokenizer()

    model = config.model.build(init_device="meta")
    if config.trainer.load_path:
        log.info("Deferring weight init to checkpoint load_path=%s", config.trainer.load_path)
        model.to_empty(device=get_default_device())
        # `to_empty` breaks weight tying (Molmo2-4B). Restore the share *before* FSDP
        # wrapping and the checkpoint load so both state-dict keys fill one parameter.
        # (Requires the stage-1 checkpoint itself to hold consistent tied weights.)
        from olmo_core.nn.vision.molmo2_loader import retie_word_embeddings

        retie_word_embeddings(model)
    else:
        _init_weights_from_hf(model, config.model)

    train_module = config.train_module.build(model)
    collator = config.collator.build()

    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    datasets, weights, dataset_names = _build_mixture(tokenizer, config)
    log.info(
        "Stage 2 packing: pack=%s pack_max_crops=%d vit_crop_microbatch=%s",
        config.pack_sequences,
        config.pack_max_crops,
        os.environ.get("VIT_CROP_MICROBATCH", "16"),
    )
    data_loader = MixtureDataLoader(
        datasets,
        weights,
        collator,
        work_dir=config.trainer.save_folder,
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        pack=config.pack_sequences,
        pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
        est_tokens_per_example=EST_TOKENS_PER_EXAMPLE,
        prefetch_workers=DATA_PREFETCH_WORKERS,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        dataset_names=dataset_names,
    )

    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OVERRIDES...]

  * dry_run: Print out the final config after applying overrides and exit.
  * launch:  Launch the script on Beaker as a batch job for training.
  * train:   Run training locally (usually under torchrun).

Examples
========

Print the config:
› python {sys.argv[0]} dry_run molmo2-stage2-debug

2-GPU debug smoke (5 steps, global batch=2×16k):
› torchrun --nproc-per-node=2 {sys.argv[0]} train smoke \\
      --trainer.max_duration.value=5 --global_batch_size=32768 \\
      --train_module.rank_microbatch_size=16384 --train_module.compile_model=false

Full image-only-v9 mixture:
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=image-only-v9

Init from HF instead of stage-1 checkpoint:
› torchrun --nproc-per-node=2 {sys.argv[0]} train smoke --trainer.load_path=null

Launch on Beaker:
› python {sys.argv[0]} launch molmo2-stage2 --launch.num_nodes=1
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        pass
    else:
        print(usage)
        sys.exit(1)
