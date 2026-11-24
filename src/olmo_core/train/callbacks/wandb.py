import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from .callback import Callback

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

log = logging.getLogger(__name__)


WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"


@dataclass
class WandBCallback(Callback):
    """
    Logs metrics to Weights & Biases from rank 0.

    .. important::
        Requires the ``wandb`` package and the environment variable ``WANDB_API_KEY``.

    .. note::
        This callback logs metrics from every single step to W&B, regardless of the value
        of :data:`Trainer.metrics_collect_interval <olmo_core.train.Trainer.metrics_collect_interval>`.
    """

    priority: ClassVar[int] = 3
    """
    Initialize before checkpointing, since pre-train checkpoint saves may flush metrics.
    """

    enabled: bool = True
    """
    Set to false to disable this callback.
    """

    name: Optional[str] = None
    """
    The name to give the W&B run.
    """

    project: Optional[str] = None
    """
    The W&B project to use.
    """

    entity: Optional[str] = None
    """
    The W&B entity to use.
    """

    group: Optional[str] = None
    """
    The W&B group to use.
    """

    tags: Optional[List[str]] = None
    """
    Tags to assign the run.
    """

    notes: Optional[str] = None
    """
    A note/description of the run.
    """

    config: Optional[Dict[str, Any]] = None
    """
    The config to load to W&B.
    """

    cancel_tags: Optional[List[str]] = field(
        default_factory=lambda: ["cancel", "canceled", "cancelled"]
    )
    """
    If you add any of these tags to a run on W&B, the run will cancel itself.
    Defaults to ``["cancel", "canceled", "cancelled"]``.
    """

    cancel_check_interval: Optional[int] = None
    """
    Check for cancel tags every this many steps. Defaults to
    :data:`olmo_core.train.Trainer.cancel_check_interval`.
    """

    _wandb = None
    _run_path = None
    _finalized: bool = False

    @property
    def wandb(self):
        if self._wandb is None:
            import wandb  # type: ignore

            self._wandb = wandb
        return self._wandb

    @property
    def run(self) -> "Run":
        return self.wandb.run

    @property
    def run_path(self):
        return self._run_path

    @property
    def finalized(self) -> bool:
        return self._finalized

    def finalize(self, exit_code: int = 0):
        if not self.finalized:
            if exit_code > 0:
                log.warning("Finalizing failed W&B run...")
            else:
                log.info("Finalizing successful W&B run...")
            # The 'quiet' argument was removed in newer wandb versions.
            self.wandb.finish(exit_code=exit_code)
            self._finalized = True

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            if WANDB_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

            self.wandb
            wandb_dir = self.trainer.work_dir / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            self.wandb.init(
                dir=wandb_dir,
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
            )
            self._run_path = self.run.path  # type: ignore

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.enabled and get_rank() == 0:
            self.wandb.log(metrics, step=step)

    def post_step(self):
        cancel_check_interval = self.cancel_check_interval or self.trainer.cancel_check_interval
        if self.enabled and get_rank() == 0 and self.step % cancel_check_interval == 0:
            self.trainer.run_bookkeeping_op(
                self.check_if_canceled,
                allow_multiple=False,
                distributed=False,
            )

    def on_error(self, exc: BaseException):
        del exc
        if self.enabled and get_rank() == 0 and self.run is not None:
            log.warning("Finalizing failed W&B run...")
            self.finalize(exit_code=1)

    def close(self):
        if self.enabled and get_rank() == 0 and self.run is not None:
            self.finalize()

    def check_if_canceled(self):
        if self.enabled and self.cancel_tags:
            from requests.exceptions import RequestException
            from wandb.errors import CommError  # type: ignore

            try:
                # NOTE: need to re-initialize the API client every time, otherwise
                # I guess it return cached run data.
                api = self.wandb.Api(api_key=os.environ[WANDB_API_KEY_ENV_VAR], timeout=5)
                run = api.run(self.run_path)  # type: ignore
                for tag in run.tags or []:
                    if tag.lower() in self.cancel_tags:
                        self.trainer.cancel_run("canceled from W&B tag")
                        return
            except (RequestException, CommError, TimeoutError):
                log.warning("Failed to communicate with W&B API")
