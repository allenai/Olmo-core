"""
Train a 32B OLMo model. Run this script without any arguments to see usage info.
"""

import logging
from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, build_config, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, DurationUnit, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    DownstreamEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 2048 * SEQUENCE_LENGTH

log = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_32B(vocab_size=common.tokenizer.padded_vocab_size())


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=2 * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=6e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            # fused=True,
            compile=True,
        ),
        compile_model=True,
        # dp_config=TransformerDataParallelConfig(
        #     name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        # ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=128 // 32,  # common.launch.num_nodes // 2,
        ),
        # ac_config=TransformerActivationCheckpointingConfig(TransformerActivationCheckpointingMode.full),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=[f"blocks.{i}.feed_forward" for i in range(64)],
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    project_name = "peteish32"
    return (
        TrainerConfig(
            save_folder=f"gs://ai2-llm/checkpoints/{project_name}/",
            checkpointer=CheckpointerConfig(
                save_thread_count=1, load_thread_count=32, throttle_uploads=True
            ),
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration(int(6.5e12), DurationUnit.tokens),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "profiler", ProfilerCallback(skip_first=3, wait=10, warmup=2, active=3, repeat=1)
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project=project_name,
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project=project_name,
                enabled=False,
                cancel_check_interval=10,
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
                tokenizer=common.tokenizer,
                eval_interval=1000,
            ),
        )
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
    )
    main(config_builder=config_builder)
