"""
Template training script. Please copy and modify and for your own needs.
Run this script without any arguments to see its usage.
"""

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
)
from olmo_core.train import (
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# TODO: update these settings for your use case

# Set this if you want to start training a new run from an existing checkpoint (like for annealing).
# NOTE: You do NOT need to set this on a restart of the same run as long as the trainer's 'save_folder' hasn't changed.
CHECKPOINT: Optional[str] = None

# Data configuration.
SEQUENCE_LENGTH = 4096
TOKENIZER_CONFIG = TokenizerConfig.dolma2()
DATA_PATHS: List[str] = []  # paths or URLs to your '.npy' tokenized training data files.
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = 8 * SEQUENCE_LENGTH
INTRA_DOCUMENT_MASKING = False

# Model and optim.
MODEL_CONFIG = TransformerConfig.olmo2_1B(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())
LEARNING_RATE = 4e-4

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 1
BEAKER_WORKSPACE = "ai2/OLMo-core"
BEAKER_BUDGET = "ai2/oe-other"

# Logging.
COMET_PROJECT: Optional[str] = None  # set this to enable Comet logging
WANDB_PROJECT: Optional[str] = None  # set this to enable W&B logging

###########################
#### END CONFIGURATION ####
###########################

# NOTE: in most cases you shouldn't need to edit below this line.


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    work_dir = get_work_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
        generate_doc_lengths=INTRA_DOCUMENT_MASKING,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=2000),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                workspace="ai2",
                project=COMET_PROJECT,
                cancel_check_interval=10,
                enabled=COMET_PROJECT is not None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity="ai2",
                project=WANDB_PROJECT,
                cancel_check_interval=10,
                enabled=WANDB_PROJECT is not None,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_recommended_evals(TOKENIZER_CONFIG, SEQUENCE_LENGTH, BEAKER_CLUSTER)
    )

    return ExperimentConfig(
        model=MODEL_CONFIG,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=build_launch_config(
            name=run_name,
            root_dir=root_dir,
            cmd=[script, "train", run_name, *overrides],
            cluster=BEAKER_CLUSTER,
            workspace=BEAKER_WORKSPACE,
            budget=BEAKER_BUDGET,
            num_nodes=NUM_NODES,
        ),
    ).merge(overrides)


def train(config: ExperimentConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Maybe load from existing checkpoint.
    if CHECKPOINT is not None and not trainer.maybe_load_checkpoint(trainer.save_folder):
        trainer.load_checkpoint(CHECKPOINT)

    # Train.
    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OVERRIDES...]

  * dry_run: Print out the final config after applying overrides and exit. Useful for debugging.
  * launch:  Launch the script on Beaker as a batch job for training.
  * train:   Run the script for training locally. This should usually not be called directly.

Examples
========

Print the config:
› python {sys.argv[0]} dry_run run01 --launch.num_nodes=2

Launch the training run as a Beaker batch job:
› python {sys.argv[0]} launch run01 --launch.num_nodes=2

Launch the training run locally (e.g. in a Beaker interactive session):
› torchrun --nproc-per-node=8 {sys.argv[0]} train run01 --launch.num_nodes=2
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        prepare_training_environment()
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
