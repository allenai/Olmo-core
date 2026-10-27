"""
Official annealing script for OLMo-2-0325-32B.
"""

import argparse
import logging
from typing import List, Tuple

from olmo_core.config import DType
from olmo_core.data import (
    DataMixBase,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
    TokenizerName,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    LinearWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import (
    ExperimentConfig,
    get_cli_parser,
    get_lr_from_checkpoint,
    main,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 4096


class AnnealingDataMix(DataMixBase):
    """
    Defines the annealing mixes. To create a new mix, make a new file in this folder and add its
    name (without the '.txt' extension) below.
    """

    dolmino100 = "dolmino100"
    dolmino300 = "dolmino300"
    jallyrun = "jallyrun"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        assert tokenizer == TokenizerName.dolma2
        paths = []
        labels = []

        with open(f"src/scripts/train/anneal/{self}.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(f"{base_dir}{line}")
                labels.append(line.split("/")[1])

        return paths, labels


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    # Starting LR should be where the checkpoint left off.
    log.info("Inferring LR from checkpoint...")
    starting_lr = get_lr_from_checkpoint(opts.checkpoint)
    log.info(f"Will start annealing from LR={starting_lr}")

    config = ExperimentConfig(
        model=TransformerConfig.olmo2_32B(vocab_size=tokenizer_config.padded_vocab_size()),
        dataset=NumpyFSLDatasetConfig.from_data_mix(
            AnnealingDataMix.dolmino100,
            tokenizer=tokenizer_config,
            mix_base_dir=opts.data_root,
            sequence_length=sequence_length,
            work_dir=opts.work_dir,
        ),
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=2048 * 4096,  # NOTE: this is specified in TOKENS, not instances.
            seed=34521,  # NOTE: can update this to change data order.
            num_workers=4,
        ),
        train_module=TransformerTrainModuleConfig(
            rank_microbatch_size=2 * 4096,  # NOTE: again this is specified in tokens.
            max_sequence_length=sequence_length,
            z_loss_multiplier=1e-5,
            compile_model=True,
            optim=SkipStepAdamWConfig(
                lr=starting_lr,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
                compile=True,
            ),
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                num_replicas=get_world_size() // 64,  # NOTE: tune this,
            ),
            ac_config=TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.selected_modules,
                modules=["blocks.*.feed_forward"],
            ),
            scheduler=LinearWithWarmup(
                units=SchedulerUnits.steps,
                warmup=0,
                alpha_f=0.0,
            ),
            max_grad_norm=1.0,
        ),
        trainer=TrainerConfig(
            save_folder=opts.save_folder,
            load_strategy=LoadStrategy.always,
            checkpointer=CheckpointerConfig(
                save_thread_count=1, load_thread_count=8, throttle_uploads=True
            ),
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(100e9)),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                enabled=False,  # NOTE: change to true to enable
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                enabled=False,  # NOTE: change to true to enable
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
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
        ),
        load_path=opts.checkpoint,
    ).merge(overrides)

    # Make sure this is an 'AnnealingDataMix' instance.
    config.dataset.mix = AnnealingDataMix(config.dataset.mix)
    return config


def _get_parser() -> argparse.ArgumentParser:
    parser = get_cli_parser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the OLMo-2-0325-32B checkpoint to load and anneal from.",
    )
    return parser


if __name__ == "__main__":
    main(build_config, parser=_get_parser())
