import logging
import time
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from olmo_core.data import (
    NumpyDatasetConfig,
    NumpyPaddedFSLDataset,
    NumpyVSLDatasetConfig,
    TextDataLoaderBase,
    TokenizerConfig,
)
from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval import Evaluator
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.utils import (
    cuda_sync_debug_mode,
    format_float,
    gc_cuda,
    get_default_device,
    move_to_device,
)

from ..common import Duration, MetricMergeStrategy
from ..train_module import (
    EvalBatchSizeUnit,
    EvalBatchSpec,
    OLMoDDPTrainModule,
    TransformerTrainModule,
)
from .callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from olmo_eval import HFTokenizer

    from ..trainer import Trainer

log = logging.getLogger(__name__)


@dataclass
class EvaluatorCallback(Callback):
    """
    Runs in-loop evaluations for a :class:`~olmo_core.train.train_module.TransformerTrainModule`
    periodically during training.
    """

    evaluators: List[Evaluator] = field(default_factory=list)
    """
    The evaluators to run.
    """

    eval_interval: Optional[int] = 1000
    """
    The interval (in steps) with which to run the evaluators.
    """

    fixed_steps: Optional[List[int]] = None
    """
    A list of fixed steps at which to run the evaluators.
    """

    eval_on_startup: bool = False
    """
    Whether to run an evaluation when the trainer starts up.
    """

    eval_on_finish: bool = False
    """
    Whether to run an evaluation when training finishes.
    """

    cancel_after_first_eval: bool = False
    """
    If ``True``, cancel the run after running evals for the first time.
    This combined with ``eval_on_startup=True`` is useful if you just want to run in-loop evals
    without training any longer.
    """

    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """
    The duration to run each evaluator for.
    """

    log_interval: int = 5
    """
    How often to log eval progress to the console during an eval loop.
    """

    def post_attach(self):
        if not isinstance(self.trainer.train_module, (TransformerTrainModule, OLMoDDPTrainModule)):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' only supports transformer train modules "
                f"('{TransformerTrainModule.__name__}', '{OLMoDDPTrainModule.__name__}')"
            )

    def pre_train(self):
        if self.eval_on_startup:
            self.perform_eval()

    def post_train(self):
        if self.eval_on_finish:
            self.perform_eval()

    def post_step(self):
        if self.step <= 1:
            return

        if (self.eval_interval is not None and self.step % self.eval_interval == 0) or (
            self.fixed_steps is not None and self.step in self.fixed_steps
        ):
            self.perform_eval()

    def perform_eval(self, prefix: str = "eval"):
        """
        Run evaluation on all evaluators and record metrics.

        :param prefix: Prefix for metric names (e.g., "eval" or "eval/merged").
            Metrics will be recorded as "{prefix}/{evaluator.name}/{metric_name}".
        """
        # Put model in eval train mode.
        # TODO: make sure grads will be zeroed at this point
        #  self.trainer.optim.zero_grad(set_to_none=True)
        #  self.trainer.model.eval()
        dp_world_size = get_world_size(self.trainer.dp_process_group)

        evaluator_times = []
        evaluator_names = []
        evaluator_bs = []

        for evaluator in self.evaluators:
            log.info(f"Running {evaluator.display_name} evals...")
            start_time = time.monotonic()
            evaluator.reset_metrics()
            eval_step = 0
            eval_tokens = 0
            for batch in evaluator:
                eval_step += 1
                eval_tokens += batch["input_ids"].numel() * dp_world_size

                batch = move_to_device(batch, get_default_device())
                with torch.no_grad():
                    # Run forward pass, get logits and un-reduced CE loss.
                    labels = get_labels(batch)
                    if "label_mask" in batch:
                        # ce_loss[:, i] is the loss for predicting token i + 1, so the mask
                        # used to select loss values must be shifted the same way as the labels.
                        batch["label_mask"] = labels != -100
                    output = self.trainer.train_module.eval_batch(batch, labels=labels)
                    assert isinstance(output, LMOutputWithLoss)
                    logits, _, ce_loss, _ = output

                    # NOTE: might have host-device syncs here but that's okay.
                    with cuda_sync_debug_mode(0):
                        evaluator.update_metrics(batch, ce_loss, logits)

                if self.eval_duration.due(step=eval_step, tokens=eval_tokens, epoch=1):
                    self._log_progress(evaluator, eval_step)
                    break

                if eval_step % self.log_interval == 0 or eval_step == evaluator.total_batches:
                    self._log_progress(evaluator, eval_step)

            # NOTE: going to have a host-device sync here but that's okay. It's only once
            # per evaluator.
            metrics_str = []
            evaluation_names = []
            with cuda_sync_debug_mode(0):
                metrics = evaluator.compute_metrics()
                for name, value in metrics.items():
                    evaluation_names.append(name)
                    metrics_str.append(f"    {name}={format_float(value.item())}")
                    self.trainer.record_metric(f"{prefix}/{evaluator.name}/{name}", value)

            evaluator_times.append(time.monotonic() - start_time)
            evaluator_names.append(evaluation_names)
            evaluator_bs.append(eval_step)

            gc_cuda()
            log.info(
                f"Finished {evaluator.display_name} evals in {time.monotonic() - start_time:.1f} seconds. Metrics:\n"
                + "\n".join(metrics_str)
            )

        # Sort by evaluator_times in ascending order
        sorted_evaluators = sorted(
            zip(evaluator_names, evaluator_bs, evaluator_times), key=lambda x: x[2]
        )

        # Record evaluation speed.
        eval_speeds = []
        for names, bs, t in sorted_evaluators:
            name = names[0]
            eval_speeds.append(f"    {name} (+variants): {t:.1f} sec ({bs} batches)")
        total_time = sum(evaluator_times)
        total_bs = sum(int(bs) if bs is not None else 0 for bs in evaluator_bs)
        eval_speeds.append(
            f"    Total evaluation time: {total_time:.1f} seconds ({total_bs} batches)"
        )
        log.info("Evaluation speed:\n" + "\n".join(eval_speeds))

        self.trainer.record_metric(
            "throughput/in-loop eval time (s)", total_time, merge_strategy=MetricMergeStrategy.sum
        )
        self.trainer.record_metric(
            "throughput/in-loop eval batches", total_bs, merge_strategy=MetricMergeStrategy.sum
        )

        if self.cancel_after_first_eval:
            self.trainer.cancel_run(
                "canceled from evaluator callback since 'cancel_after_first_eval' is set",
                no_sync=True,  # 'no_sync' because we're calling this from all ranks at the same time.
            )

    def _log_progress(self, evaluator: Evaluator, eval_step: int):
        if evaluator.total_batches is not None:
            log.info(f"[eval={evaluator.name},step={eval_step}/{evaluator.total_batches}]")
        else:
            log.info(f"[eval={evaluator.name},step={eval_step}]")


@dataclass
class LMEvaluatorCallbackConfig(CallbackConfig):
    eval_dataset: NumpyDatasetConfig
    eval_interval: Optional[int] = 1000
    fixed_steps: Optional[List[int]] = None
    eval_on_startup: bool = False
    eval_on_finish: bool = False
    cancel_after_first_eval: bool = False
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5
    deterministic: bool = True
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        dataset_max_sequence_length: int
        if isinstance(self.eval_dataset, NumpyVSLDatasetConfig):
            dataset_max_sequence_length = self.eval_dataset.max_sequence_length
        else:
            assert hasattr(self.eval_dataset, "sequence_length")
            dataset_max_sequence_length = self.eval_dataset.sequence_length  # type: ignore

        batch_spec = trainer.train_module.eval_batch_spec
        if (
            batch_spec.max_sequence_length is not None
            and dataset_max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for the LM eval dataset ({dataset_max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        global_eval_batch_size: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            global_eval_batch_size = batch_spec.rank_batch_size * get_world_size(
                trainer.dp_process_group
            )
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            global_eval_batch_size = (
                batch_spec.rank_batch_size
                * dataset_max_sequence_length
                * get_world_size(trainer.dp_process_group)
            )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        dataset = self.eval_dataset.build()
        if not isinstance(dataset, NumpyPaddedFSLDataset):
            raise OLMoConfigurationError(
                f"Expected a padded FSL dataset, got '{dataset.__class__.__name__}' instead"
            )

        if not isinstance(trainer.data_loader, TextDataLoaderBase):
            raise OLMoConfigurationError(
                f"Expected a text-based data loader, got '{dataset.__class__.__name__}' instead"
            )

        evaluator = LMEvaluator.from_numpy_dataset(
            dataset,
            name="lm",
            global_batch_size=global_eval_batch_size,
            collator=trainer.data_loader.collator,
            device=trainer.device,
            dp_process_group=trainer.dp_process_group,
            deterministic=self.deterministic,
        )
        return EvaluatorCallback(
            evaluators=[evaluator],
            eval_interval=self.eval_interval,
            fixed_steps=self.fixed_steps,
            log_interval=self.log_interval,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            eval_duration=self.eval_duration,
            eval_on_finish=self.eval_on_finish,
        )


@cache
def _all_tasks() -> Set[str]:
    from olmo_eval import list_tasks

    return set(list_tasks())


class DownstreamEvaluator(Evaluator):
    metric_type_to_label = {
        "f1_v1": "F1 score",
        "acc_v1": "accuracy",
        "len_norm_v1": "length-normalized accuracy",
        "pmi_dc_v1": "PMI-DC accuracy",
        "ce_loss_v1": "CE loss",
        "bpb_v1": "BPB",
        "soft_v1": "soft loss",
        "soft_log_v1": "log soft loss",
        "f1_v2": "F1 score v2",
        "acc_v2": "accuracy v2",
        "len_norm_v2": "length-normalized accuracy v2",
        "pmi_dc_v2": "PMI-DC accuracy v2",
        "ce_loss_v2": "CE loss v2",
        "bpb_v2": "BPB v2",
        "soft_v2": "soft loss v2",
        "soft_log_v2": "log soft loss v2",
    }

    def __init__(
        self,
        *,
        name: str,
        task: str,
        batch_spec: EvalBatchSpec,
        tokenizer: "HFTokenizer",
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        lazy: bool = False,
    ):
        from olmo_eval import ICLMetric

        if task not in _all_tasks():
            raise OLMoConfigurationError(f"Unknown downstream eval task: '{task}'")

        self.label = task
        self.batch_spec = batch_spec
        self.tokenizer = tokenizer
        self.device = device  # set here for _build_data_loader() to use
        self.dp_process_group = dp_process_group
        self.metric: Optional[ICLMetric] = None
        if lazy:
            log.info(f"Initializing lazy DownstreamEvaluator for task '{self.label}'")

        super().__init__(
            name=name,
            batches=None if lazy else self._build_data_loader(),
            batches_factory=self._build_data_loader if lazy else None,
            device=device,
        )

    @property
    def display_name(self) -> str:
        return f"{self.name} '{self.label}'"

    def _build_data_loader(self) -> DataLoader:
        from olmo_eval import ICLMetric, ICLMultiChoiceTaskDataset, build_task

        log.info(f"Building downstream eval task dataset for '{self.label}'...")

        task_dataset: ICLMultiChoiceTaskDataset
        if self.batch_spec.fixed_sequence_length:
            assert self.batch_spec.max_sequence_length is not None
            task_dataset = build_task(
                self.label,
                self.tokenizer,
                model_ctx_len=self.batch_spec.max_sequence_length,
                fixed_ctx_len=True,
            )
        elif self.batch_spec.max_sequence_length is not None:
            task_dataset = build_task(
                self.label, self.tokenizer, model_ctx_len=self.batch_spec.max_sequence_length
            )
        else:
            task_dataset = build_task(self.label, self.tokenizer)

        self.metric = ICLMetric(metric_type=task_dataset.metric_type).to(
            self.device or get_default_device()
        )
        sampler: Optional[DistributedSampler] = None
        if is_distributed():
            sampler = DistributedSampler(
                task_dataset,  # type: ignore
                drop_last=False,
                shuffle=False,
                num_replicas=get_world_size(self.dp_process_group),
                rank=get_rank(self.dp_process_group),
            )

        if (
            self.batch_spec.max_sequence_length is not None
            and task_dataset.max_sequence_length > self.batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for downstream eval task '{self.label}' ({task_dataset.max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({self.batch_spec.max_sequence_length:,d} tokens)"
            )

        rank_batch_size_instances: int
        if self.batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            rank_batch_size_instances = self.batch_spec.rank_batch_size
        elif self.batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            if self.batch_spec.fixed_sequence_length:
                assert self.batch_spec.max_sequence_length is not None
                if self.batch_spec.rank_batch_size % self.batch_spec.max_sequence_length != 0:
                    raise OLMoConfigurationError(
                        f"The eval batch size ({self.batch_spec.rank_batch_size} tokens) must be divisible "
                        f"by the maximum eval sequence length ({self.batch_spec.max_sequence_length:,d} tokens)"
                    )
                rank_batch_size_instances = (
                    self.batch_spec.rank_batch_size // self.batch_spec.max_sequence_length
                )
            else:
                rank_batch_size_instances = (
                    self.batch_spec.rank_batch_size // task_dataset.max_sequence_length
                )
        else:
            raise NotImplementedError(self.batch_spec.batch_size_unit)

        log.info(
            f"Using per-rank batch size of {rank_batch_size_instances} instances "
            f"for downstream eval task '{self.label}' with max sequence length {task_dataset.max_sequence_length:,d} tokens"
        )

        return DataLoader(
            task_dataset,  # type: ignore
            batch_size=rank_batch_size_instances,
            collate_fn=task_dataset.collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
        )

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: Optional[torch.Tensor], logits: Optional[torch.Tensor]
    ) -> None:
        del ce_loss
        assert self.metric is not None
        if logits is None:
            raise RuntimeError(
                "Downstream evaluators require full logits, but logits are None. "
                "This happens when context parallelism (CP) or tensor parallelism (TP) is enabled. "
                "Please disable downstream evals when using CP or TP."
            )
        self.metric.update(batch, logits)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        assert self.metric is not None
        metric_type_to_value = self.metric.compute()
        outputs = {}
        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label[metric_type]})"
            outputs[key] = value
        return outputs

    def reset_metrics(self) -> None:
        if self.metric is not None:
            self.metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    tasks: List[str]
    tokenizer: TokenizerConfig
    eval_interval: Optional[int] = 1000
    fixed_steps: Optional[List[int]] = None
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    eval_on_startup: bool = False
    eval_on_finish: bool = False
    cancel_after_first_eval: bool = False
    log_interval: int = 5
    lazy: bool = False
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        from olmo_eval import HFTokenizer

        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Tokenizer 'identifier' required to build a concrete tokenizer"
            )

        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        evaluators: List[Evaluator] = []
        for task in sorted(self.tasks):
            evaluators.append(
                DownstreamEvaluator(
                    name="downstream",
                    task=task,
                    batch_spec=trainer.train_module.eval_batch_spec,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                    lazy=self.lazy,
                )
            )

        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            fixed_steps=self.fixed_steps,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
            eval_on_finish=self.eval_on_finish,
        )
