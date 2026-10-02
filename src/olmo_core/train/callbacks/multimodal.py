"""Callbacks for multimodal training: model initialization, held-out response-loss evaluation,
and the bookkeeping variants (W&B resume, same-step metric snapshots) the alignment pipeline
relies on. Everything here subclasses the shared callbacks without changing them.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch.distributed as dist
from torch.utils.data import Subset

from olmo_core.aliases import PathOrStr
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.data_loader import MultimodalDataLoader
from olmo_core.distributed.utils import broadcast_object, get_rank, get_world_size
from olmo_core.eval import Evaluator
from olmo_core.eval.multimodal_image_pairing import (
    MultimodalImagePairDataset,
    build_bounded_image_pairs,
)
from olmo_core.eval.multimodal_lm_evaluator import (
    MultimodalBlankImageEvaluator,
    MultimodalLMEvaluator,
)
from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError
from olmo_core.io import file_exists, is_url, normalize_path, resource_path, upload
from olmo_core.train.common import Duration

from .beaker import BeakerCallback
from .callback import Callback, CallbackConfig
from .checkpointer import CheckpointerCallback, CheckpointRemovalStrategy
from .evaluator_callback import EvaluatorCallback
from .metric_saver import MetricSaverCallback
from .wandb import WANDB_API_KEY_ENV_VAR, WandBCallback

if TYPE_CHECKING:
    from olmo_core.train import Trainer

log = logging.getLogger(__name__)


def write_file_overwrite(
    trainer: "Trainer", name: str, contents: str | bytes, dir: Optional[PathOrStr] = None
) -> PathOrStr:
    """
    Write a file into the trainer's save folder (or ``dir``), replacing an existing one.

    :meth:`~olmo_core.train.Trainer.write_file` follows the checkpointer's overwrite policy, which
    the alignment pipeline keeps at ``save_overwrite=False`` to protect checkpoints; metric
    snapshots and completion markers are rewritten deliberately, so they bypass it here.

    :returns: The path/URL of the file.
    """
    target_dir = normalize_path(dir if dir is not None else trainer.save_folder)
    name = normalize_path(name)
    data = contents.encode() if isinstance(contents, str) else contents
    if is_url(target_dir):
        target = f"{target_dir}/{name}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / Path(name).name
            tmp_path.write_bytes(data)
            upload(tmp_path, target, save_overwrite=True)
        return target
    path = Path(target_dir) / name
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp_file = tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False)
    try:
        tmp_file.write(data)
        tmp_file.close()
        os.replace(tmp_file.name, path)
    finally:
        Path(tmp_file.name).unlink(missing_ok=True)
    return path


@dataclass
class MultimodalMetricSaverCallback(MetricSaverCallback):
    """
    A :class:`~olmo_core.train.callbacks.MetricSaverCallback` whose per-step snapshots may be
    rewritten: training, checkpointing and evaluation can log separate fragments at the same step
    (and a resumed run re-logs the step it restarts from), so the file is replaced each time
    while checkpoint overwrite protection stays in force.
    """

    def _write_metrics(self, fname: str, metrics: Dict[str, float]) -> PathOrStr:
        return write_file_overwrite(self.trainer, fname, json.dumps(metrics))


@dataclass
class RestoreMetricsCallback(Callback):
    """Preserve existing metrics when a resumed run only needs to finish evaluation.

    Runs before startup evaluation can overwrite a metric saver's step snapshot. Only
    the restored checkpoint's exact step is loaded; later, unsaved training is excluded.
    Pair it with :class:`MultimodalMetricSaverCallback`, which rewrites the snapshot.
    """

    metrics_callback: str
    """Name of the attached :class:`MetricSaverCallback` receiving the saved metrics."""

    priority: ClassVar[int] = 4

    def pre_train(self):
        """Merge the existing same-step file into the metric saver on rank zero."""
        if get_rank() != 0 or not self.trainer.checkpoint_loaded:
            return
        callback = self.trainer.callbacks[self.metrics_callback]
        filename = callback.step_metrics_fname.format(step=self.step)
        if file_exists(f"{self.trainer.save_folder}/{filename}"):
            path = resource_path(self.trainer.save_folder, filename)
            callback.log_metrics(self.step, json.loads(path.read_text()))


@dataclass
class MultimodalEvaluatorCallback(EvaluatorCallback):
    """
    An :class:`~olmo_core.train.callbacks.EvaluatorCallback` that remembers the last step it
    evaluated under the ``eval`` prefix, so ``eval_on_finish`` does not evaluate a step that the
    interval (or startup) evaluation just covered. The memory is cleared when training starts and
    when a checkpoint is loaded.
    """

    _last_eval_step: Optional[int] = field(default=None, init=False, repr=False, compare=False)

    def pre_train(self):
        self._last_eval_step = None
        super().pre_train()

    def post_checkpoint_loaded(self, path: PathOrStr):
        """Invalidate the evaluation memory after loading model weights."""
        del path
        self._last_eval_step = None

    def post_train(self):
        if self.eval_on_finish and self._last_eval_step != self.step:
            self.perform_eval()

    def perform_eval(self, prefix: str = "eval"):
        super().perform_eval(prefix=prefix)
        if prefix == "eval":
            self._last_eval_step = self.step


@dataclass
class MultimodalCheckpointerCallback(CheckpointerCallback):
    """
    A :class:`~olmo_core.train.callbacks.CheckpointerCallback` that, on start, also takes over
    the permanent checkpoints an earlier run of the same job left behind (steps up to the current
    one) and trims them to ``max_checkpoints``, so a resumed alignment phase keeps the same
    bounded set of checkpoints as an uninterrupted one. With ``remove=never`` nothing is removed.
    """

    def pre_train(self):
        super().pre_train()
        if not self.enabled or self.remove == CheckpointRemovalStrategy.never:
            return

        permanent_checkpoints: List[Tuple[int, str]] = []
        # Only search from rank 0 to avoid hammering remote file stores with requests.
        if get_rank() == 0:
            try:
                for step_num, path in self.checkpointer.find_checkpoints(
                    self.save_folder, ephemeral=False
                ):
                    if step_num <= self.step:
                        permanent_checkpoints.append((step_num, path))
            except FileNotFoundError:
                pass
        permanent_checkpoints = broadcast_object(permanent_checkpoints)

        # Keep a checkpoint saved above (e.g. the pre-train one) even if discovery has not seen it
        # yet, without duplicating one it did see.
        steps_by_path = {path: step for step, path in permanent_checkpoints}
        for path in self._checkpoints:
            steps_by_path.setdefault(path, self.step)
        self._checkpoints = [
            path for _, path in sorted((step, path) for path, step in steps_by_path.items())
        ]
        self._trim_checkpoints()


@dataclass
class MultimodalWandBCallback(WandBCallback):
    """
    A :class:`~olmo_core.train.callbacks.WandBCallback` that can resume the same W&B run after a
    restart: the run ID is stored in the trainer checkpoint and reused when the run identity
    (name, project, entity) matches.
    """

    auto_resume: bool = False
    """Persist the W&B run ID in trainer checkpoints and resume the same run after a restart."""

    run_id: Optional[str] = None
    """An existing W&B run ID to resume, e.g. for a checkpoint saved before the ID was stored."""

    _resume_step: Optional[int] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "name": self.name,
            "project": self.project,
            "entity": self.entity,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        identity = (self.name, self.project, self.entity)
        saved_identity = (
            state_dict.get("name"),
            state_dict.get("project"),
            state_dict.get("entity"),
        )
        if self.auto_resume and identity == saved_identity:
            if (run_id := state_dict.get("run_id")) is not None:
                self.run_id = run_id
                self._resume_step = state_dict.get("step")

    def pre_train(self):
        if not (self.enabled and get_rank() == 0):
            return
        if WANDB_API_KEY_ENV_VAR not in os.environ:
            raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

        self.wandb
        wandb_dir = self.trainer.work_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs: Dict[str, Any] = {}
        if self.auto_resume and self.run_id is not None:
            resume_step = self._resume_step
            if resume_step is None and getattr(self.trainer, "checkpoint_loaded", False):
                resume_step = self.step
            if resume_step is not None:
                log.info(
                    "Resuming W&B run '%s' for training checkpoint step %s",
                    self.run_id,
                    resume_step,
                )
            else:
                log.info("Resuming W&B run '%s'", self.run_id)
            init_kwargs.update(id=self.run_id, resume="allow", allow_val_change=True)

        self.wandb.init(
            dir=wandb_dir,
            project=self.project,
            entity=self.entity,
            group=self.group,
            name=self.name,
            tags=self.tags,
            notes=self.notes,
            config=self.config,
            **init_kwargs,
        )
        self.run_id = self.run.id
        self._run_path = self.run.path  # type: ignore


@dataclass
class MultimodalBeakerCallback(BeakerCallback):
    """
    A :class:`~olmo_core.train.callbacks.BeakerCallback` for resumed alignment runs.

    :class:`MultimodalWandBCallback` resumes the same W&B run after a restart, whose config still
    holds the previous Beaker experiment's URL and id; the base callback's plain
    ``config.update`` then raises on the changed values. The keys are written first with
    ``allow_val_change`` so the base update is a no-op.
    """

    def _experiment_url(self) -> str:
        from olmo_core.launch.beaker import get_beaker_client, get_beaker_experiment_id

        if self.experiment_id is None:
            self.experiment_id = get_beaker_experiment_id()
        assert self.experiment_id is not None
        with get_beaker_client() as beaker:
            return beaker.workload.url(beaker.workload.get(self.experiment_id))

    def _preset_wandb_config(self) -> None:
        url: Optional[str] = None
        for callback in self.trainer.callbacks.values():
            if (
                isinstance(callback, WandBCallback)
                and callback.enabled
                and callback.run is not None
            ):
                if url is None:
                    url = self._experiment_url()
                callback.run.config.update(
                    {"beaker_experiment_url": url, "beaker_experiment_id": self.experiment_id},
                    allow_val_change=True,
                )

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            self._preset_wandb_config()
        super().pre_train()


@dataclass
class InitializeMultimodalModelCallback(Callback):
    """Initialize a fresh alignment run after the trainer has attempted a resume.

    A restored checkpoint, including a step-zero checkpoint, always takes precedence.
    Language weights are loaded without optimizer state; the vision loader and image-token
    initializer synchronize their weights with any optimizer master parameters.

    This callback currently supports :class:`~olmo_core.train.train_module.transformer.multimodal_train_module.MultimodalOLMoDDPTrainModule`.
    """

    priority: ClassVar[int] = 4
    """Initialize weights before evaluators and pre-training checkpoint saves."""

    language_checkpoint: str
    """Language-model checkpoint directory, or its ``model_and_optim`` subdirectory."""

    vision_model_id: str
    """Hugging Face repository containing a SigLIP vision encoder."""

    image_token_ids: list[int]
    """Input-embedding rows to initialize for image tokens."""

    vision_revision: str | None = None
    cache_dir: str | None = None
    seed: int = 12536
    load_threads: int | None = None

    def pre_train(self):
        """Load pretrained components only when no full checkpoint was restored."""
        if self.trainer.checkpoint_loaded:
            return

        from olmo_core.nn.vision import load_siglip_hf_vision_state_dict
        from olmo_core.train.train_module.transformer.multimodal_train_module import (
            MultimodalOLMoDDPTrainModule,
        )

        train_module = self.trainer.train_module
        if not isinstance(train_module, MultimodalOLMoDDPTrainModule):
            raise OLMoConfigurationError(
                "Separate language/vision initialization currently requires "
                "MultimodalOLMoDDPTrainModule"
            )

        state_dir = self.language_checkpoint.rstrip("/")
        if state_dir.rsplit("/", 1)[-1] != "model_and_optim":
            state_dir += "/model_and_optim"
        log.info("Loading pretrained language weights from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=self.load_threads,
            load_optim_state=False,
        )

        log.info("Loading pretrained vision weights from %s", self.vision_model_id)
        vision_state = load_siglip_hf_vision_state_dict(
            self.vision_model_id, revision=self.vision_revision, cache_dir=self.cache_dir
        )
        train_module.load_siglip_vision_state_dict(vision_state)
        train_module.assert_vision_optimizer_state_synced()
        if self.image_token_ids:
            train_module.reset_image_token_rows(
                self.image_token_ids, seed=self.seed, reset_output_rows=False
            )


@dataclass
class MultimodalEvaluatorCallbackConfig(CallbackConfig):
    """Build deterministic response-loss evaluators from explicit held-out sources.

    Each source evaluates the first ``examples_per_source`` rows in its configured order.
    Provide a prepared selection when a subset or stratified holdout is required. Mixture
    loss weights and calibration means are not used for evaluation.
    """

    eval_dataset: MultimodalMixtureConfig
    """Held-out datasets; these must be disjoint from the training sources."""

    sequence_length: int
    rank_batch_size: int = 1
    """Number of evaluation sequences per data-parallel rank."""

    examples_per_source: int = 512
    eval_interval: int | None = 500
    blank_image_sources: list[str] = field(default_factory=list)
    matched_image_sources: list[str] = field(default_factory=list)
    """Sources with paired correct/wrong-image diagnostics, separate from the full validation."""

    matched_image_examples: int = 64
    matched_image_candidates: int = 512
    """Maximum held-out rows to preprocess once when selecting exact-geometry image pairs."""

    early_response_tokens: int | None = 8
    """Also compare the first N supervised positions per paired sequence; None disables this."""

    eval_on_startup: bool = True
    eval_on_finish: bool = True
    cancel_after_first_eval: bool = False
    """Cancel after the first evaluation; combine with startup evaluation for an eval-only run."""

    seed: int = 0

    def build(self, trainer: "Trainer") -> MultimodalEvaluatorCallback:
        """Build source evaluators and bounded paired image-content diagnostics."""
        if self.sequence_length <= 0 or self.rank_batch_size <= 0:
            raise OLMoConfigurationError("Evaluation sequence and batch sizes must be positive")
        if self.examples_per_source <= 0:
            raise OLMoConfigurationError("examples_per_source must be positive")
        if self.eval_interval is not None and self.eval_interval <= 0:
            raise OLMoConfigurationError("eval_interval must be positive or None")
        unknown_sources = set(self.blank_image_sources) - set(self.eval_dataset.sources)
        if unknown_sources:
            raise OLMoConfigurationError(
                f"Blank-image controls refer to unknown sources: {sorted(unknown_sources)}"
            )
        unknown_sources = set(self.matched_image_sources) - set(self.eval_dataset.sources)
        if unknown_sources:
            raise OLMoConfigurationError(
                f"Matched-image controls refer to unknown sources: {sorted(unknown_sources)}"
            )
        if self.early_response_tokens is not None and self.early_response_tokens <= 0:
            raise OLMoConfigurationError("early_response_tokens must be positive or None")
        if self.matched_image_sources and (
            self.matched_image_examples <= 0
            or self.matched_image_candidates < self.matched_image_examples
            or self.seed < 0
        ):
            raise OLMoConfigurationError("Invalid matched-image example/candidate count or seed")
        dp_world_size = get_world_size(trainer.dp_process_group)
        dp_rank = get_rank(trainer.dp_process_group)
        global_instances = self.rank_batch_size * dp_world_size
        if self.examples_per_source % global_instances:
            raise OLMoConfigurationError(
                "examples_per_source must be divisible by the global evaluation batch size"
            )
        if self.matched_image_sources and self.matched_image_examples % global_instances:
            raise OLMoConfigurationError(
                "matched_image_examples must be divisible by the global evaluation batch size"
            )

        tokenizer, token_ids = self.eval_dataset.build_tokenizer()
        sources = self.eval_dataset.build_sources(tokenizer, token_ids)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise OLMoConfigurationError("The tokenizer must define a pad or EOS token")
        collator = MultimodalCollator(
            pad_token_id=pad_token_id,
            pad_sequence_length=self.sequence_length,
            batch_metadata=True,
        )

        def build_loader(dataset, name):
            return MultimodalDataLoader(
                dataset,
                collator,
                work_dir=trainer.work_dir / name,
                global_batch_size=global_instances * self.sequence_length,
                seed=self.seed,
                shuffle=False,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
            )

        evaluators: list[Evaluator] = []
        for name, dataset in sources.items():
            if len(dataset) < self.examples_per_source:
                raise OLMoConfigurationError(
                    f"Evaluation source {name!r} has {len(dataset)} examples, "
                    f"but {self.examples_per_source} were requested"
                )
            loader = build_loader(
                Subset(dataset, range(self.examples_per_source)), f"{name}_validation"
            )
            evaluators.append(
                MultimodalLMEvaluator(
                    name=f"{name}-validation",
                    batches=loader,
                    device=trainer.device,
                    process_group=trainer.dp_process_group,
                    deterministic=True,
                )
            )
            if name in self.blank_image_sources:
                evaluators.append(
                    MultimodalBlankImageEvaluator(
                        name=f"{name}-blank-image",
                        batches=loader,
                        device=trainer.device,
                        process_group=trainer.dp_process_group,
                        deterministic=True,
                    )
                )
            if name in self.matched_image_sources:
                # Prepare only on the DP leader, then share small index pairs (or the error).
                # This avoids preprocessing the same candidate images on every GPU rank.
                result: dict[str, Any] | None = None
                if dp_rank == 0 or not dist.is_initialized():
                    try:
                        result = {
                            "pairs": build_bounded_image_pairs(
                                dataset,
                                examples=self.matched_image_examples,
                                max_candidates=self.matched_image_candidates,
                                seed=self.seed,
                            )
                        }
                    except Exception as error:  # noqa: BLE001 - propagate failures to every rank
                        result = {"error": f"{type(error).__name__}: {error}"}
                group = trainer.dp_process_group
                leader = dist.get_global_rank(group, 0) if group is not None else 0
                result = broadcast_object(result, src=leader, group=group)
                assert result is not None
                if "error" in result:
                    raise OLMoConfigurationError(f"Image pairing for {name!r}: {result['error']}")
                pairs = result["pairs"]
                log.info(
                    "Matched-image evaluation %s: %d pairs from at most %d held-out candidates",
                    name,
                    len(pairs),
                    min(self.matched_image_candidates, len(dataset)),
                )
                correct_loader = build_loader(
                    MultimodalImagePairDataset(dataset, pairs, wrong_images=False),
                    f"{name}_matched_correct",
                )
                wrong_loader = build_loader(
                    MultimodalImagePairDataset(dataset, pairs, wrong_images=True),
                    f"{name}_matched_wrong",
                )
                prefixes: list[int | None] = [None]
                if self.early_response_tokens is not None:
                    prefixes.append(self.early_response_tokens)
                for prefix in prefixes:
                    suffix = "" if prefix is None else f"-first{prefix}"
                    correct = MultimodalLMEvaluator(
                        name=f"{name}-matched-correct{suffix}",
                        batches=correct_loader,
                        device=trainer.device,
                        process_group=group,
                        response_prefix_tokens=prefix,
                    )
                    wrong = MultimodalLMEvaluator(
                        name=f"{name}-matched-wrong{suffix}",
                        batches=wrong_loader,
                        device=trainer.device,
                        process_group=group,
                        response_prefix_tokens=prefix,
                        reference_evaluator=correct,
                    )
                    evaluators.extend([correct, wrong])
        max_examples = max(
            self.examples_per_source,
            self.matched_image_examples if self.matched_image_sources else 0,
        )
        return MultimodalEvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            eval_duration=Duration.steps(max_examples // global_instances),
            eval_on_startup=self.eval_on_startup,
            eval_on_finish=self.eval_on_finish,
            cancel_after_first_eval=self.cancel_after_first_eval,
        )
