"""
Official pre-training script for OLMo-3-1025-7B.

Part 2 of 2. See OLMo-3-1025-7B-pretrain-1.py for part 1.
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
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    HalfCosWithWarmup,
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
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8192 * 512  # ~4M tokens
LR = 3e-4


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # pad to a multiple of 128
        attn_backend=AttentionBackendName.flash_2,
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_official,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=8,
    )

    # Scheduler settings from the first half
    original_warmup_steps = 2000
    original_max_steps = int(5e12) // GLOBAL_BATCH_SIZE

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 8192,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=HalfCosWithWarmup(  # Scheduler updated to extend lr from where we left off.
            units=SchedulerUnits.steps, warmup=original_max_steps // 2 + original_warmup_steps // 2
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
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            load_path="https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step596047/",
            load_strategy=LoadStrategy.always,
            load_trainer_state=True,
            load_optim_state=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(7e12)),  # Changed from 5T -> 7T
            hard_stop=Duration.epochs(1),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
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
                eval_interval=10_000,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(
                    [  # "fast" task set
                        # Subset of OLMES
                        "arc_challenge_test_bpb_5shot",
                        "arc_challenge_test_mc_5shot_fast",
                        "arc_easy_test_bpb_5shot",
                        "arc_easy_test_mc_5shot_fast",
                        "hellaswag_bpb_5shot",
                        "mmlu_humanities_test_bpb_5shot",
                        "mmlu_humanities_test_mc_5shot_fast",
                        "mmlu_other_test_bpb_5shot",
                        "mmlu_other_test_mc_5shot_fast",
                        "mmlu_social_sciences_test_bpb_5shot",
                        "mmlu_social_sciences_test_mc_5shot_fast",
                        "mmlu_stem_test_bpb_5shot",
                        "mmlu_stem_test_mc_5shot_fast",
                        # Basic Skills
                        "basic_skills_arithmetic_rc_5shot",
                        "basic_skills_coding_rc_5shot",
                        "basic_skills_common_knowledge_rc_5shot",
                        "basic_skills_logical_reasoning_rc_5shot",
                        "basic_skills_pattern_rc_5shot",
                        "basic_skills_string_operations_rc_5shot",
                        # Gen tasks BPB
                        "codex_humaneval_gold_bpb_3shot",
                        "codex_mbpp_gold_bpb_3shot",
                        "minerva_math_500_gold_bpb_0shot",
                        "mt_mbpp_cpp_gold_bpb_3shot",
                        "mt_mbpp_java_gold_bpb_3shot",
                        "mt_mbpp_rust_gold_bpb_3shot",
                        # Sanity check for MCQA ability
                        "copycolors_10way_fast",
                    ]
                ),
                tokenizer=tokenizer_config,
                eval_interval=10_000,
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
