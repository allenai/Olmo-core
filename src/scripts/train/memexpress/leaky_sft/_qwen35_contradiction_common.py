"""LeakySFT: six context-loss weights, otherwise identical contradiction-only runs.

Contrast: alpha=0/.1/.3/.5/.7/1 on the dense Qwen3.5-4B hybrid CPT checkpoint.
Readout: held-out contradiction F1 by context length, plus output/context CE.
Data-controlled: one loader epoch, same seed, BFD packing and retained documents.
Not compute-matched: nonzero alpha backpropagates context loss; zero only logs it.
This is a new two-GPU pilot, not a reproduction of the multi-task or 256K runs.
"""

from dataclasses import replace

from sweep_settings import ARMS, DATA_ROOT, GPUS, SEQUENCE_LENGTH

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LongDocStrategy,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CliContext, ExperimentConfig, SubCmd
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEED = 34521
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/qwen35-4b-dense-longmino/step2385/model_and_optim"
SAVE_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/amandab"
WORK_DIR = "/weka/oe-training-default/amandab/leakysft/xlong256k-filter32k-cache"


def build_leaky_experiment(context: CliContext, *, arm: str) -> ExperimentConfig:
    """Build a two-GPU arm; use launch_sweep.py for minimum-runtime enforcement."""
    if context.cmd in (SubCmd.launch, SubCmd.launch_prep):
        raise ValueError(
            "Use launch_sweep.py: it enforces two GPUs, Jupiter, urgent, min-runtime=1h"
        )
    alpha = ARMS[arm]
    tokenizer = TokenizerConfig.qwen3_5()
    doc_tokenizer = replace(tokenizer, bos_token_id=None)
    model = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    model.lm_head.loss_implementation = LMLossImplementation.fused_linear
    source = NumpyDocumentSourceConfig(
        source_paths=[f"{DATA_ROOT}/token_ids_part_*.npy"],
        label_mask_paths=[f"{DATA_ROOT}/labels_mask_*.npy"],
        tokenizer=doc_tokenizer,
        expand_glob=True,
    )
    config = ExperimentConfig(
        run_name=context.run_name,
        launch=None,  # minimum runtime is absent from the legacy BeakerLaunchConfig API
        init_seed=SEED,
        model=model,
        dataset=[
            PackingInstanceSourceConfig(
                sources=[source],
                tokenizer=doc_tokenizer,
                sequence_length=SEQUENCE_LENGTH,
                long_doc_strategy=LongDocStrategy.exclude,
                source_group_size=1000000,
                label="contradiction",
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            tokenizer=doc_tokenizer,
            work_dir=WORK_DIR,
            seed=SEED,
            global_batch_size=GPUS * SEQUENCE_LENGTH,
            num_workers=0,
            generate_doc_lengths=True,
        ),
        train_module=TransformerTrainModuleConfig(
            cpt_loss_scaler=alpha,
            rank_microbatch_size=SEQUENCE_LENGTH,
            max_sequence_length=SEQUENCE_LENGTH,
            optim=SkipStepAdamWConfig(
                lr=1e-5,
                weight_decay=0.0,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
            ),
            scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
            compile_model=False,
            # FSDP shards FP32 parameters, gradients and Adam state across two GPUs.
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                shard_degree=GPUS,
                num_replicas=1,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            ),
            cp_config=None,
            tp_config=None,
            pp_config=None,
            ac_config=TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.full
            ),
            float8_config=Float8Config(enabled=False),
            z_loss_multiplier=None,
            max_grad_norm=1.0,
        ),
        trainer=TrainerConfig(
            save_folder=f"{SAVE_ROOT}/{context.run_name}",
            save_overwrite=False,
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_optim_state=False,
            load_trainer_state=False,
            max_duration=Duration.epochs(1),
            cancel_check_interval=1,
            metrics_collect_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=250,
                ephemeral_save_interval=100,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=context.run_name,
                group="LeakySFT-contradiction-xlong-filter32k-1ep",
                entity="ai2-llm",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=1,
            ),
        ),
    )
    # Keep the selected data, filtering, budget and parallelism fixed across the sweep.
    config = config.merge(context.overrides)
    if (
        config.train_module.cpt_loss_scaler != alpha
        or config.trainer.max_duration != Duration.epochs(1)
        or config.data_loader.global_batch_size != GPUS * SEQUENCE_LENGTH
        or config.train_module.max_sequence_length != SEQUENCE_LENGTH
        or config.train_module.rank_microbatch_size != SEQUENCE_LENGTH
        or config.train_module.cp_config is not None
        or config.train_module.tp_config is not None
        or config.train_module.pp_config is not None
        or config.train_module.dp_config.shard_degree != GPUS
        or config.dataset[0].sequence_length != SEQUENCE_LENGTH
        or config.dataset[0].long_doc_strategy != LongDocStrategy.exclude
        or config.dataset[0].sources != [source]
    ):
        raise ValueError(
            "LeakySFT sweep requires the selected alpha, one epoch, two GPUs and 32K windows on the filtered 256K corpus"
        )
    return config
