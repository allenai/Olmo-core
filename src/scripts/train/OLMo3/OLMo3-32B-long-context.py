from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig
from olmo_core.data.numpy_dataset import InstanceFilterConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import (
    LinearWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024  # ~8M tokens, 2**23
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.0002071235285  # same as midtraining


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo3_32B(
        vocab_size=common.tokenizer.padded_vocab_size(),
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = SEQUENCE_LENGTH
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=8,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.3
        ),
        cp_config=TransformerContextParallelConfig.llama3(
            degree=8,  # 64k tokens per instance -> 8k tokens per device
            head_stride=4,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10
    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%z')}"

    return (
        TrainerConfig(
            load_path="gs://ai2-llm/checkpoints/stego32-midtraining-runs-merged-step23842-resharded16",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            save_folder=f"gs://ai2-llm/checkpoints/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_TOKENS),
            hard_stop=None,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="olmo-cookbook",
                enabled=False,  # enable via CLI
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name, enabled=False),  # enable via CLI
        )
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = True,
    include_instance_filter: bool = False,
) -> DataComponents:
    """
    Default dataset and data loader configurations. Constructs a simple FSL dataset and data loader
    configuration with default settings.
    """
    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=intra_document_masking,  # enables intra-document masking
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        # Preserve the original training environment. Run this historical script from its
        # corresponding Olmo-core revision, whose dependencies match this image.
        beaker_image=OLMoCoreBeakerImage.tch270_cu128,
        include_instance_filter=True,
        intra_document_masking=True,
        include_default_evals=False,
    )

    main(config_builder=config_builder)
