"""
Example of how to train a Llama transformer language model with LayerNorm Scaling (LNS) enabled.

This script is identical to the standard Llama example except that it lives in a separate
namespace so you can easily run it side-by-side with a baseline run.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama_lns/train.py run_name [OVERRIDES...]
"""

import os
import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 1024

# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
DATA_ROOT = os.environ.get("OLMO_DATA_ROOT", "http://olmo-data.org/examples/c4-en/gpt2").rstrip("/")
DATA_PATHS = [
    f"{DATA_ROOT}/c4-train.00000-00099.npy",
    f"{DATA_ROOT}/c4-train.00100-00199.npy",
    f"{DATA_ROOT}/c4-train.00200-00299.npy",
    f"{DATA_ROOT}/c4-train.00300-00399.npy",
    f"{DATA_ROOT}/c4-train.00400-00499.npy",
    f"{DATA_ROOT}/c4-train.00500-00599.npy",
    f"{DATA_ROOT}/c4-train.00600-00699.npy",
    f"{DATA_ROOT}/c4-train.00700-00799.npy",
    f"{DATA_ROOT}/c4-train.00800-00899.npy",
    f"{DATA_ROOT}/c4-train.00900-00999.npy",
    f"{DATA_ROOT}/c4-train.01000-01023.npy",
]
EVAL_DATA_PATHS = [f"{DATA_ROOT}/c4-validation.00000-00008.npy"]
DATA_WORK_DIR = "/tmp/dataset-cache"


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.llama2_271M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    # Select the LayerNorm-Scaled transformer block implementation.
    assert isinstance(model_config.block, TransformerBlockConfig)
    model_config.block.name = TransformerBlockType.default_scaled

    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
        tokenizer=tokenizer_config,
        work_dir=DATA_WORK_DIR,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * SEQUENCE_LENGTH,
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=100),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=EVAL_DATA_PATHS,
                    metadata=[{"label": "c4-validation"}],
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=DATA_WORK_DIR,
                ),
                eval_interval=250,
                eval_duration=Duration.steps(50),
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=["hellaswag"],
                tokenizer=tokenizer_config,
                eval_interval=250,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

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

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    main(run_name, overrides=overrides)
    teardown_training_environment()
