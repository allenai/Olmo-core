from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import (
    build_launch_config,
    get_gpu_type,
    get_root_dir,
    get_work_dir,
)
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
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
    TransformerTrainModuleConfig,
)

SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**19  # ~524k tokens
CHINCHILLA_MULTIPLE = 1.0  # Train to 1x Chinchilla optimality
FOR_BENCHMARKING = False


def estimate_lr(model_params: int, chinchilla_multiple: float = 1.0) -> float:
    # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
    # Optimized for 1x Chinchilla.
    lr = 0.0047 * (model_params / 108_000_000) ** (-1 / 3)
    # Scale down for longer training (sqrt scaling heuristic)
    lr /= chinchilla_multiple**0.5
    return lr


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    gpu_type = get_gpu_type(cli_context.cluster)
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/OLMo_3",
        num_nodes=1,
        nccl_debug=False,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()

    attn_backend = (
        AttentionBackendName.flash_4 if "B200" in gpu_type else AttentionBackendName.flash_3
    )
    model_config = TransformerConfig.olmo3_190M(
        vocab_size=tokenizer_config.padded_vocab_size(), attn_backend=attn_backend
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH * 2,
        max_sequence_length=SEQ_LENGTH,
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
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=2000),
    )

    intra_document_masking = False
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,  # note: updated to 0925 mix
        mix_base_dir="gs://ai2-llm",
        work_dir=work_dir,
        tokenizer=tokenizer_config,
        sequence_length=SEQ_LENGTH,
        # max target sequence length doesn't affect how the data is loaded, just how it's cached behind the scenes
        max_target_sequence_length=max(SEQ_LENGTH, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=34521, num_workers=8
    )

    cancel_check_interval = 20
    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            metrics_collect_interval=20,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.chinchilla_tokens(
                CHINCHILLA_MULTIPLE, model_params=model_config.num_active_non_embedding_params
            ),
        )
        .with_callbacks(configure_required_callbacks(run_name_with_ts))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=True,
                enabled=not FOR_BENCHMARKING,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=False,  # --trainer.callbacks.wandb.enabled=True to enable
                cancel_check_interval=cancel_check_interval,
            ),
        )
    )
    if not FOR_BENCHMARKING:
        trainer_config = trainer_config.with_recommended_evals(
            tokenizer_config, SEQ_LENGTH, cli_context.cluster, task_set="fast"
        )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=1337,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/OLMo3/OLMo-3-190M.py dry_run debug_run ai2/jupiter

        To launch a training run on Jupiter w/ 2 nodes:
        python src/scripts/train/OLMo3/OLMo-3-190M.py launch my_run ai2/jupiter \
            --launch.num_nodes=2 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
