from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import (
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 8 * 1024
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens

# Remove heads to match params/TPS of OLMo3 7B transformer. This is to enable a
# fair comparison with OLMo3 7B. If training from scratch, we recommend setting the
# number of attention heads to 32 (or some power of 2 that makes sense for your model size).
REMOVE_HEADS = 2

DATA_MIX = DataMix.OLMo_mix_0925
MAX_DURATION = Duration.epochs(1)
INSTANCE_FILTER = True


def build_model_config(common: CommonComponents) -> TransformerConfig:
    """
    Build OLMo 3.2 7B hybrid model config.

    This is a hybrid architecture combining Gated Delta Net (GDN) recurrent layers
    with standard attention layers in a 3:1 ratio (3 GDN layers followed by 1 attention
    layer, repeating). The model is based on OLMo3 7B but with reduced heads to match
    params/TPS for fair comparison with the pure transformer variant.
    """
    config = TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, AttentionConfig)

    # Remove heads (and scale down d_model) to compensate for extra params.
    config.d_model -= REMOVE_HEADS * 128
    num_heads = config.block.sequence_mixer.n_heads - REMOVE_HEADS
    config.block.sequence_mixer.n_heads = num_heads
    assert config.d_model / num_heads == 128

    attn_block = config.block

    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=num_heads,
            head_dim=int(0.75 * config.d_model / num_heads),
            allow_neg_eigval=True,
        ),
    )

    # 3 GDN layers followed by 1 attention layer, repeating.
    config.block = {"gdn": gdn_block, "attn": attn_block}
    config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    assert config.n_layers % len(config.block_pattern) == 0

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
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
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=2000),
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DATA_MIX,
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
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


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            save_folder=f"{common.root_dir}/checkpoints/willm/linear-rnns/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=MAX_DURATION,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="linear-rnns",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_recommended_evals(common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast")
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=INSTANCE_FILTER,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
