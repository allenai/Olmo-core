import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Optional, cast

import torch
import torch.distributed as dist
from rich import print

from olmo_core.config import Config, StrEnum
from olmo_core.data import (
    DataLoaderBase,
    DataLoaderConfig,
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    InstanceSource,
    InstanceSourceConfig,
)
from olmo_core.data.numpy_dataset import NumpyFSLDatasetConfig
from olmo_core.distributed.utils import get_local_rank
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GAPMonitorCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SlackNotifierCallback,
)
from olmo_core.train.train_module import TrainModuleConfig
from olmo_core.utils import prepare_cli_environment, seed_all

from .common import build_launch_config, get_beaker_username, get_root_dir, get_work_dir

log = logging.getLogger(__name__)


@dataclass
class CliContext(Config):
    script: str
    cmd: "SubCmd"
    run_name: str
    cluster: str
    overrides: List[str]

    @property
    def remote_cmd(self) -> List[str]:
        return [
            self.script,
            self.cmd.post_launch_subcmd(),
            self.run_name,
            self.cluster,
            *self.overrides,
        ]


@dataclass
class CommonComponents(Config):
    run_name: str

    root_dir: str
    work_dir: str
    save_folder: str

    launch: Optional[BeakerLaunchConfig]

    tokenizer: TokenizerConfig
    max_sequence_length: int
    global_batch_size: int


@dataclass
class DataComponents(Config):
    dataset: NumpyDatasetConfig | List[InstanceSourceConfig]
    data_loader: DataLoaderConfig


@dataclass
class ExperimentConfig(Config):
    run_name: str
    launch: Optional[BeakerLaunchConfig]
    model: TransformerConfig
    dataset: NumpyDatasetConfig | List[InstanceSourceConfig]
    data_loader: DataLoaderConfig
    train_module: TrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536
    backend: Optional[str] = "cpu:gloo,cuda:nccl"
    process_group_timeout_seconds: float = 15 * 60


class SubCmd(StrEnum):
    launch = "launch"
    train = "train"
    train_single = "train_single"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"
    eval_checkpoints = "eval_checkpoints"

    def post_launch_subcmd(self) -> "SubCmd":
        if self in (SubCmd.launch_prep, SubCmd.prep):
            return SubCmd.prep
        else:
            return SubCmd.train

    def prepare_environment(self, config: ExperimentConfig):
        if self in (SubCmd.launch, SubCmd.dry_run, SubCmd.prep, SubCmd.launch_prep):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment(
                backend=config.backend,
                timeout=timedelta(seconds=config.process_group_timeout_seconds),
            )
        elif self == SubCmd.train_single:
            prepare_training_environment(backend=None)
        elif self == SubCmd.eval_checkpoints:
            prepare_training_environment(
                backend=config.backend,
                timeout=timedelta(seconds=config.process_group_timeout_seconds),
            )
        else:
            raise NotImplementedError(self)

    def run(self, config: ExperimentConfig):
        if get_local_rank() == 0:
            print(config)
            print(
                "\n"
                f"[b blue]Total parameters:[/]         {config.model.num_params:,d} ({config.model.num_active_params:,d} active)\n"
                f"[b blue]Non-embedding parameters:[/] {config.model.num_non_embedding_params:,d} ({config.model.num_active_non_embedding_params:,d} active)"
            )

        if self == SubCmd.launch:
            launch(config)
        elif self == SubCmd.dry_run:
            pass
        elif self == SubCmd.train:
            train(config)
            teardown_training_environment()
        elif self == SubCmd.train_single:
            if (dp_config := getattr(config.train_module, "dp_config", None)) is not None:
                log.warning(
                    "'dp_config' is set to %s, but you can't use data parallelism when running on a single node. Disabling.",
                    dp_config,
                )
                config.train_module.dp_config = None  # type: ignore
            if (tp_config := getattr(config.train_module, "tp_config", None)) is not None:
                log.warning(
                    "'tp_config' is set to %s, but you can't use tensor parallelism when running on a single node. Disabling.",
                    tp_config,
                )
                config.train_module.tp_config = None  # type: ignore
            train(config)
            teardown_training_environment()
        elif self == SubCmd.prep:
            prep(config)
        elif self == SubCmd.launch_prep:
            launch_prep(config)
        elif self == SubCmd.eval_checkpoints:
            eval_checkpoints(config)
            teardown_training_environment()
        else:
            raise NotImplementedError(self)


ConfigBuilder = Callable[[CliContext], ExperimentConfig]
"""
Type alias for a function that builds an ExperimentConfig based on a CliContext.
"""


def build_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
    num_execution_units: Optional[int] = None,
    flight_recorder: bool = False,
) -> CommonComponents:
    root_dir = get_root_dir(cli_context.cluster)
    beaker_user = get_beaker_username()

    launch_config: Optional[BeakerLaunchConfig] = None
    if beaker_user is not None:
        cmd_to_launch = cli_context.cmd.post_launch_subcmd()
        launch_config = build_launch_config(
            name=f"{cli_context.run_name}-{cmd_to_launch}",
            root_dir=root_dir,
            cmd=cli_context.remote_cmd,
            cluster=cli_context.cluster,
            nccl_debug=True,
            flight_recorder=flight_recorder,
            beaker_image=beaker_image,
            num_nodes=num_nodes,
            workspace=beaker_workspace,
            num_execution_units=num_execution_units,
        )
        launch_config.launch_timeout = 5 * 60

    if beaker_user is not None:
        save_folder = f"{root_dir}/checkpoints/{beaker_user.lower()}/{cli_context.run_name}"
    else:
        save_folder = f"{root_dir}/checkpoints/{cli_context.run_name}"

    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
        launch=launch_config,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def build_default_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    """
    Default dataset and data loader configurations. Constructs a simple FSL dataset and data loader
    configuration with default settings.
    """
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        # max target sequence length doesn't affect how the data is loaded, just how it's cached behind the scenes
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=4
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def _build_required_callbacks(common: CommonComponents) -> Dict[str, Callback]:
    callbacks = {
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=common.run_name, enabled=False),
        "gap_monitor": GAPMonitorCallback(enabled=False),
    }
    if common.launch is not None:
        callbacks["beaker"] = BeakerCallback()
    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()
    return callbacks


def _build_default_eval_callbacks(common: CommonComponents) -> Dict[str, Callback]:
    return {
        "lm_evaluator": LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                mix_base_dir=common.root_dir,
                sequence_length=common.max_sequence_length,
                tokenizer=common.tokenizer,
                work_dir=common.work_dir,
            ),
            eval_interval=1000,
        ),
        "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
            tasks=["hellaswag"],
            tokenizer=common.tokenizer,
            eval_interval=1000,
        ),
    }


def build_config(
    cli_context: CliContext,
    *,
    common_config_builder: Callable[..., CommonComponents] = build_common_components,
    data_config_builder: Callable[..., DataComponents] = build_default_data_components,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    train_module_config_builder: Callable[[CommonComponents], TrainModuleConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    tokenizer: TokenizerConfig = TokenizerConfig.dolma2(),
    global_batch_size: int,
    max_sequence_length: int = 4096,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
    #  use_hostname_constraints: bool = False,
    flight_recorder: bool = False,
    num_execution_units: Optional[int] = None,
    include_default_evals: bool = False,
    process_group_timeout_seconds: float = 15 * 60,
    **data_kwargs,
) -> ExperimentConfig:
    """
    Build an ``ExperimentConfig`` from a CLI context with good defaults. Allows for easy
    customization of individual components of the configuration.

    :param cli_context: The CLI context containing run name and overrides.
    :param common_config_builder: Function to build common components. This should accept a
        ``CliContext`` instance and return a ``CommonComponents`` instance.
    :param data_config_builder: Function to build data components. This should accept a
        ``CommonComponents`` instance and return a ``DataComponents`` instance.
    :param model_config_builder: Function to build the transformer model configuration. This should accept a
        ``CommonComponents`` instance and return a ``TransformerConfig`` instance.
    :param train_module_config_builder: Function to build the training module configuration. This should accept a
        ``CommonComponents`` instance and return a ``TrainModuleConfig`` instance.
    :param trainer_config_builder: Function to build the trainer configuration. This should accept a
        ``CommonComponents`` instance and return a ``TrainerConfig`` instance.
    :param finalize_config: Optional function to finalize the configuration. This should accept an
        ``ExperimentConfig`` instance and modify it in place.
    :param tokenizer: The tokenizer configuration to use.
    :param global_batch_size: The global batch size for training.
    :param max_sequence_length: Maximum sequence length for the model. This is typically used to set
        the sequence length for the dataset and data loader, as well as the maximum sequence length
        for the model.
    :param beaker_image: The Beaker image to use for the experiment.
    :param num_nodes: Number of nodes to use for training.
    :param beaker_workspace: The Beaker workspace to use.
    :param num_execution_units: Number of execution units for Beaker.
    :param include_default_evals: Whether to include default evaluation callbacks.
    :param process_group_timeout_seconds: Distributed collective timeout, including the wait
        for rank-zero dataset preparation. Increase this for cold caches or run ``prep`` first.
    :param data_kwargs: Additional keyword arguments to pass to the data config builder.
    :returns: The complete ``ExperimentConfig``.
    """

    common = common_config_builder(
        cli_context,
        tokenizer=tokenizer,
        global_batch_size=global_batch_size,
        max_sequence_length=max_sequence_length,
        beaker_image=beaker_image,
        num_nodes=num_nodes,
        beaker_workspace=beaker_workspace,
        #  use_hostname_constraints=use_hostname_constraints,
        flight_recorder=flight_recorder,
        num_execution_units=num_execution_units,
    )

    data = data_config_builder(common, **data_kwargs)

    model = model_config_builder(common)

    trainer = trainer_config_builder(common)
    callbacks_to_add = _build_required_callbacks(common)
    if include_default_evals:
        default_evals = _build_default_eval_callbacks(common)
        callbacks_to_add.update(default_evals)
    for name, cb in callbacks_to_add.items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    train_module = train_module_config_builder(common)

    config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=common.launch,
        model=model,
        dataset=data.dataset,
        data_loader=data.data_loader,
        train_module=train_module,
        trainer=trainer,
        process_group_timeout_seconds=process_group_timeout_seconds,
    )

    config = config.merge(cli_context.overrides)
    if finalize_config is not None:
        finalize_config(config)

    return config


def launch(config: ExperimentConfig):
    assert config.launch is not None
    config.launch.launch()


def _build_data_loader(
    config: ExperimentConfig, dp_process_group: Optional[dist.ProcessGroup] = None
) -> DataLoaderBase:
    if isinstance(config.data_loader, NumpyDataLoaderConfig):
        assert isinstance(config.dataset, NumpyDatasetConfig)
        dataset = config.dataset.build()
        return config.data_loader.build(dataset, dp_process_group=dp_process_group)
    elif isinstance(config.data_loader, ComposableDataLoaderConfig):
        assert isinstance(config.dataset, list)
        work_dir = config.data_loader.work_dir
        assert work_dir is not None, "Please set 'work_dir' on data loader config"
        sources: List[InstanceSource] = []
        for source in config.dataset:
            assert isinstance(source, InstanceSourceConfig)
            sources.append(source.build(work_dir))
        return config.data_loader.build(*sources, dp_process_group=dp_process_group)
    else:
        raise NotImplementedError(type(config.data_loader))


def launch_prep(config: ExperimentConfig):
    assert config.launch is not None
    config.launch.num_gpus = 0
    config.launch.num_nodes = 1
    config.launch.launch(torchrun=False)


def prep(config: ExperimentConfig):
    data_loader = _build_data_loader(config)
    data_loader.reshuffle(epoch=1)


def train(config: ExperimentConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    data_loader = _build_data_loader(config, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train (also handles checkpoint loading)
    trainer.fit()


def eval_checkpoints(config: ExperimentConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components in eval-only mode (no optimizer, no DP wrapping).
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model, eval_only=True)
    data_loader = _build_data_loader(config, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader, eval_only=True)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Eval (also handles checkpoint loading)
    trainer.eval_checkpoints()


def main(*, config_builder: ConfigBuilder) -> None:
    """
    Main entry point for running Ai2-internal experiments.

    This function handles command-line argument parsing and executes the appropriate
    subcommand (launch, train, prep, etc.) based on user input.

    :param config_builder: A function that builds an ExperimentConfig from command-line
        arguments. It should accept (script, cmd, run_name, cluster, overrides)
        and return an ExperimentConfig instance.

    .. tip::
        Use ``functools.partial`` + ``build_config()`` to assemble a function that can be used as
        a config builder without needing to construct an entire ``ExperimentConfig``. For example::

            from functools import partial
            from olmo_core.internal.experiment import build_config, ConfigBuilder

            config_builder: ConfigBuilder = partial(
                build_config,
                data_config_builder=my_data_config_builder(),
                model_config_builder=my_model_config_builder(),
                train_module_config_builder=my_train_module_config_builder(),
                trainer_config_builder=my_trainer_config_builder(),
            )
            main(config_builder=config_builder)

    """

    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]prep:[/]        Prepare the dataset ahead of training to save GPU time.
[b magenta]launch_prep:[/] Launch the script on Beaker with the [b magenta]prep[/] subcommand.
[b magenta]dry_run:[/]     Pretty print the config and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} run01 ai2/neptune --launch.num_nodes=2[/]
    """.strip()

    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv
    cmd = SubCmd(cmd)
    cli_context = CliContext(script, cmd, run_name, cluster, overrides)

    config: ExperimentConfig = config_builder(cli_context)
    cmd.prepare_environment(config)
    cmd.run(config)
