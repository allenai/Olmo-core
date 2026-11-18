import dataclasses
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.launch.utils import is_running_in_beaker_batch_job

from .callback import Callback
from .comet import CometCallback
from .wandb import WandBCallback

log = logging.getLogger(__name__)


BEAKER_EXPERIMENT_ID_ENV_VAR = "BEAKER_EXPERIMENT_ID"
BEAKER_RESULT_DIR = "/results"


@dataclass
class BeakerCallback(Callback):
    """
    Adds metadata to the Beaker experiment description when running as a Beaker batch job.
    """

    priority: ClassVar[int] = min(CometCallback.priority - 1, WandBCallback.priority - 1)
    experiment_id: Optional[str] = None
    update_interval: Optional[int] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    """
    A JSON-serializable config to save to the results dataset as ``config.json``.
    """
    result_dir: str = BEAKER_RESULT_DIR
    """
    The directory of the Beaker results dataset where the config and other data will be saved.
    """

    _url: str | None = dataclasses.field(repr=False, default=None)
    _last_update: float | None = dataclasses.field(repr=False, default=None)

    def post_attach(self):
        if self.enabled is None:
            self.enabled = is_running_in_beaker_batch_job()

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            from olmo_core.launch.beaker import get_beaker_client

            if self.experiment_id is None:
                from olmo_core.launch.beaker import get_beaker_experiment_id

                self.experiment_id = get_beaker_experiment_id()

            assert self.experiment_id is not None
            with get_beaker_client() as beaker:
                workload = beaker.workload.get(self.experiment_id)
                beaker_url = beaker.workload.url(workload)
            log.info(f"Running in Beaker workload {beaker_url}")

            # Add Beaker URL to W&B and Comet config if available.
            for callback in self.trainer.callbacks.values():
                if isinstance(callback, WandBCallback):
                    if callback.enabled and callback.run is not None:
                        callback.run.config.update(
                            {
                                "beaker_experiment_url": beaker_url,
                                "beaker_experiment_id": self.experiment_id,
                            }
                        )
                        log.info(f"Added beaker_experiment_url to W&B config: {beaker_url}")
                        log.info(f"Added beaker_experiment_id to W&B config: {self.experiment_id}")
                elif isinstance(callback, CometCallback):
                    if callback.enabled and callback.exp is not None:
                        callback.exp.log_parameter("beaker_experiment_url", beaker_url)
                        callback.exp.log_parameter("beaker_experiment_id", self.experiment_id)
                        log.info(f"Added beaker_experiment_url to Comet: {beaker_url}")
                        log.info(f"Added beaker_experiment_id to Comet: {self.experiment_id}")

            # Ensure result dataset directory exists.
            result_dir = Path(self.result_dir) / "olmo-core"
            result_dir.mkdir(parents=True, exist_ok=True)

            # Save config to result dir.
            if self.config is not None:
                config_path = result_dir / "config.json"
                with config_path.open("w") as config_file:
                    log.info(f"Saving config to '{config_path}'")
                    json.dump(self.config, config_file)

            # Try saving Python requirements.
            requirements_path = result_dir / "requirements.txt"
            try:
                with requirements_path.open("w") as requirements_file:
                    requirements_file.write(f"# python={platform.python_version()}\n")
                with requirements_path.open("a") as requirements_file:
                    subprocess.call(
                        ["pip", "freeze"],
                        stdout=requirements_file,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
            except Exception as e:
                log.exception(f"Error saving Python packages: {e}")

            # Try to get W&B/Comet URL of experiment.
            for callback in self.trainer.callbacks.values():
                if isinstance(callback, WandBCallback) and callback.enabled:
                    if (url := callback.run.get_url()) is not None:
                        self._url = url
                    break
                elif isinstance(callback, CometCallback) and callback.enabled:
                    if (url := callback.exp.url) is not None:
                        self._url = url
                    break

            self._update()

    def post_step(self):
        update_interval = self.update_interval or self.trainer.metrics_collect_interval
        if self.enabled and get_rank() == 0 and self.step % update_interval == 0:
            # Make sure we don't update too frequently.
            if self._last_update is None or (time.monotonic() - self._last_update) > 10:
                self._update()

    def post_train(self):
        if self.enabled and get_rank() == 0:
            self._update()

    def _update(self):
        self.trainer.run_bookkeeping_op(
            self._set_description,
            op_name="beaker_set_description",
            allow_multiple=False,
            distributed=False,
        )
        self._last_update = time.monotonic()

    def _set_description(self):
        from beaker.exceptions import BeakerError, HTTPError, RequestException, RpcError
        from gantry.api import update_workload_description

        from olmo_core.launch.beaker import get_beaker_client

        description = f"[{self.trainer.training_progress}] "

        if self.description is not None:
            description = f"{description}{self.description}\n"

        if self._url is not None:
            description = f"{description}{self._url} "

        try:
            with get_beaker_client() as beaker:
                update_workload_description(description.strip(), client=beaker)
        except (RequestException, BeakerError, HTTPError, RpcError) as e:
            log.warning(f"Failed to update Beaker experiment description: {e}")
