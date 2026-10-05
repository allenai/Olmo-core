"""
Trainer :class:`Callback` implementations.
"""

from .batch_size_scheduler import BatchSizeSchedulerCallback
from .beaker import BeakerCallback
from .callback import Callback, CallbackConfig
from .checkpoint_ready_notifier import CheckpointReadyNotifierCallback
from .checkpointer import CheckpointerCallback, CheckpointRemovalStrategy
from .comet import CometCallback, CometNotificationSetting
from .config_saver import ConfigSaverCallback
from .console_logger import ConsoleLoggerCallback
from .evaluator_callback import (
    DownstreamEvaluatorCallbackConfig,
    EvaluatorCallback,
    LMEvaluatorCallbackConfig,
)
from .gap_monitor import GAPMonitorCallback
from .garbage_collector import GarbageCollectorCallback
from .gpu_memory_monitor import GPUMemoryMonitorCallback
from .hf_converter import HFConverterCallback
from .list_checkpointer import ListCheckpointerCallback
from .metric_saver import MetricSaverCallback
from .model_merger import ModelMergeCallback
from .monkey_patcher import MonkeyPatcherCallback
from .profiler import (
    NvidiaProfilerCallback,
    ProfilerCallback,
    TorchMemoryHistoryCallback,
)
from .sequence_length_scheduler import SequenceLengthSchedulerCallback
from .slack_notifier import SlackNotificationSetting, SlackNotifierCallback
from .speed_monitor import SpeedMonitorCallback
from .stability_monitor import StabilityMonitorCallback
from .unsharded_export import UnshardedModelExportCallback
from .wandb import WandBCallback

__all__ = [
    "Callback",
    "CallbackConfig",
    "CheckpointerCallback",
    "CheckpointRemovalStrategy",
    "CheckpointReadyNotifierCallback",
    "CometCallback",
    "CometNotificationSetting",
    "ConfigSaverCallback",
    "ConsoleLoggerCallback",
    "EvaluatorCallback",
    "LMEvaluatorCallbackConfig",
    "DownstreamEvaluatorCallbackConfig",
    "GAPMonitorCallback",
    "GarbageCollectorCallback",
    "GPUMemoryMonitorCallback",
    "HFConverterCallback",
    "UnshardedModelExportCallback",
    "ProfilerCallback",
    "NvidiaProfilerCallback",
    "TorchMemoryHistoryCallback",
    "SlackNotifierCallback",
    "SlackNotificationSetting",
    "SequenceLengthSchedulerCallback",
    "SpeedMonitorCallback",
    "StabilityMonitorCallback",
    "WandBCallback",
    "BeakerCallback",
    "BatchSizeSchedulerCallback",
    "MonkeyPatcherCallback",
    "MetricSaverCallback",
    "ModelMergeCallback",
    "ListCheckpointerCallback",
]

__doc__ += "\n"
for name in __all__[2:]:
    if name.endswith("Callback"):
        __doc__ += f"- :class:`{name}`\n"
