"""
Official pre-training script for OLMo-2-0325-32B.
"""

import argparse
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import (
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 4096


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo2_32B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=2048 * 4096,
        seed=34521,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * 4096,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=6e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
            foreach=False,
        ),
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=2000),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=get_world_size() // 64,  # NOTE: tune this
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            TransformerActivationCheckpointingMode.full
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # If you have 1024 GPUs, you can run slightly faster with a different config.
    if get_world_size() >= 1024:
        train_module_config.rank_microbatch_size //= 2
        train_module_config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(6.5e12)),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=200,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=opts.data_root,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=1000,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    # MMLU for backwards compatibility
                    "mmlu_stem_mc_5shot",
                    "mmlu_humanities_mc_5shot",
                    "mmlu_social_sciences_mc_5shot",
                    "mmlu_other_mc_5shot",
                    # MMLU test
                    "mmlu_stem_mc_5shot_test",
                    "mmlu_humanities_mc_5shot_test",
                    "mmlu_social_sciences_mc_5shot_test",
                    "mmlu_other_mc_5shot_test",
                    ## Core 12 tasks for backwards compatibility
                    # "arc_challenge",
                    # "arc_easy",
                    # "basic_arithmetic",
                    # "boolq",
                    # "commonsense_qa",
                    # "copa",
                    # "hellaswag",
                    # "openbook_qa",
                    # "piqa",
                    # "sciq",
                    # "social_iqa",
                    # "winogrande",
                    ## Core 12 tasks 5-shot
                    # "arc_challenge_rc_5shot",
                    # "arc_easy_rc_5shot",
                    ## "basic_arithmetic_rc_5shot",  # doesn't exist
                    ## "boolq_rc_5shot",  # we don't like it
                    # "csqa_rc_5shot",
                    ## "copa_rc_5shot",  # doesn't exist
                    # "hellaswag_rc_5shot",
                    # "openbookqa_rc_5shot",
                    # "piqa_rc_5shot",
                    ## "sciq_rc_5shot",  # doesn't exist
                    # "socialiqa_rc_5shot",
                    # "winogrande_rc_5shot",
                    ## New in-loop evals
                    # "arc_challenge_val_rc_5shot",
                    # "arc_challenge_val_mc_5shot",
                    "arc_challenge_test_rc_5shot",
                    # "arc_challenge_test_mc_5shot",
                    # "arc_easy_val_rc_5shot",
                    # "arc_easy_val_mc_5shot",
                    "arc_easy_test_rc_5shot",
                    # "arc_easy_test_mc_5shot",
                    # "boolq_val_rc_5shot",
                    # "boolq_val_mc_5shot",
                    "csqa_val_rc_5shot",
                    # "csqa_val_mc_5shot",
                    "hellaswag_val_rc_5shot",
                    # "hellaswag_val_mc_5shot",
                    # "openbookqa_val_rc_5shot",
                    # "openbookqa_val_mc_5shot",
                    "openbookqa_test_rc_5shot",
                    # "openbookqa_test_mc_5shot",
                    "piqa_val_rc_5shot",
                    # "piqa_val_mc_5shot",
                    "socialiqa_val_rc_5shot",
                    # "socialiqa_val_mc_5shot",
                    # "winogrande_val_rc_5shot",
                    # "winogrande_val_mc_5shot",
                    # "mmlu_stem_val_rc_5shot",
                    # "mmlu_stem_val_mc_5shot",
                    # "mmlu_humanities_val_rc_5shot",
                    # "mmlu_humanities_val_mc_5shot",
                    # "mmlu_social_sciences_val_rc_5shot",
                    # "mmlu_social_sciences_val_mc_5shot",
                    # "mmlu_other_val_rc_5shot",
                    # "mmlu_other_val_mc_5shot",
                ],
                tokenizer=tokenizer_config,
                eval_interval=1000,
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


if __name__ == "__main__":
    main(build_config)
