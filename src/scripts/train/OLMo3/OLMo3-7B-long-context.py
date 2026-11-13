from datetime import datetime
from typing import Optional

from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.io import join_path
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

SEQ_LENGTH = 65536
GLOBAL_BATCH_SIZE = 2**22  # ~4M tokens
MAX_TOKENS = 50_000_000_000  # 50B
LR = 0.00020712352850360292
SEED = 4123


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        # Preserve the original training environment. Run this historical script from its
        # corresponding Olmo-core revision, whose dependencies match this image.
        beaker_image="petew/olmo-core-tch270cu128",
        workspace="ai2/long-contexts",
        num_nodes=32,
        nccl_debug=True,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size()
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH,
        learning_rate=LR,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        float8_enabled=True,
        activation_memory_budget=0.7,
        cp_degree=8,
        dp_shard_degree=1,
    )

    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        str(
            join_path(
                root_dir, "preprocessed/tylerr/lc-reshard-final/v0.6/allenai/dolma2-tokenizer/*.npy"
            )
        ),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
        generate_doc_lengths=True,  # enables intra-document masking
        source_group_size=8,
        source_permutation_seed=123,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path=str(
            join_path(
                root_dir,
                "checkpoints/allysone/anneal-round5-100B-olmo25_7b-anneal-6T-decon-sparkle-motion-8730626c/step47684",
            )
        ),
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=Duration.tokens(MAX_TOKENS),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    )
    trainer_config.add_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts, wandb_group_name=cli_context.run_name
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/OLMo3/OLMo3-7B-long-context.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta w/ 8 nodes:
        python src/scripts/train/OLMo3/OLMo3-7B-long-context.py launch my_run ai2/augusta \
            --launch.num_nodes=8 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
