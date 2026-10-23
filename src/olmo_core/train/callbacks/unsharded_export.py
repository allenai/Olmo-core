"""
Callback for writing an unsharded copy of the model weights into every permanent checkpoint.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List

from cached_path import cached_path

from olmo_core.distributed.checkpoint import load_unsharded_model_state
from olmo_core.distributed.utils import get_rank
from olmo_core.io import file_exists, is_url, join_path, upload

from ..checkpoint import Checkpointer, CheckpointMetadata
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class UnshardedModelExportCallback(Callback):
    """
    Writes an unsharded copy of the model weights into every permanent checkpoint directory, next
    to the sharded ``model_and_optim/`` state:

    * ``model.safetensors``: the full model state dict, keyed as ``model.state_dict()``, with no
      optimizer state.
    * ``olmo_core_config.json``: a copy of the checkpoint's ``config.json``.

    Evaluation and inference then read one file sequentially instead of reassembling the weights
    from every training rank's shards, which is slow, degrades badly when many jobs load the same
    checkpoint at once, and reads the optimizer state for nothing. olmo-eval's ``olmo_core_vlm``
    provider loads this pair in preference to the sharded checkpoint when a directory has both.

    The export runs on rank 0 in a background bookkeeping thread and re-reads the checkpoint that
    was just saved, so it records exactly that step's weights (also with async checkpointing) and
    uses no collectives. Ephemeral checkpoints are skipped. It adds the size of the model weights,
    in their training dtype, to each permanent checkpoint.

    .. important::
        The config is copied from ``config.json`` in the checkpoint, which the
        :class:`ConfigSaverCallback` writes. Without it nothing is exported.
    """

    # Run after the ConfigSaverCallback (priority 0) has written 'config.json'.
    priority: ClassVar[int] = -1

    MODEL_FNAME: ClassVar[str] = "model.safetensors"
    CONFIG_FNAME: ClassVar[str] = "olmo_core_config.json"

    enabled: bool = True

    config_fname: str = "config.json"
    """
    The name of the config file the :class:`ConfigSaverCallback` writes to each checkpoint.
    """

    _pending: List[str] = field(default_factory=list)

    def post_checkpoint_saved(self, path):
        if not self.enabled or get_rank() != 0:
            return
        # With async checkpointing this runs on the saving thread, so the export is only queued
        # here and submitted from the main thread.
        self._pending.append(str(path))

    def post_step(self):
        self._submit_pending()

    def post_train(self):
        # Picks up the final checkpoint, which the CheckpointerCallback saves in its own
        # 'post_train'. The trainer waits for bookkeeping ops before it exits.
        self._submit_pending()

    def _submit_pending(self):
        while self._pending:
            path = self._pending.pop(0)
            self.trainer.run_bookkeeping_op(
                self.export,
                path,
                op_name="unsharded_model_export",
                distributed=False,
            )

    def export(self, path: str):
        """
        Write ``model.safetensors`` and ``olmo_core_config.json`` into the checkpoint at ``path``,
        unless it is ephemeral or has no config.
        """
        metadata_path = join_path(path, Checkpointer.METADATA_FNAME)
        if CheckpointMetadata.from_file(cached_path(metadata_path, quiet=True)).ephemeral:
            return

        config_path = join_path(path, self.config_fname)
        if not file_exists(config_path):
            log.warning(f"No '{self.config_fname}' in '{path}', not exporting unsharded weights")
            return

        log.info(f"Exporting unsharded model weights to '{path}'...")
        state_dict = load_unsharded_model_state(join_path(path, "model_and_optim"))
        self._write_safetensors(state_dict, path)
        del state_dict
        # Written last: a directory with both files counts as exported.
        config = Path(cached_path(config_path, quiet=True)).read_text()
        self.trainer.checkpointer.write_file(path, self.CONFIG_FNAME, config)
        log.info(f"Unsharded model weights exported to '{path}'")

    def _write_safetensors(self, state_dict, path: str):
        from safetensors.torch import save_file

        # Write to a temporary file first so a reader never sees a partial file.
        tmp_dir = None if is_url(path) else path
        fd, tmp_name = tempfile.mkstemp(suffix=".safetensors", dir=tmp_dir)
        os.close(fd)
        try:
            save_file(state_dict, tmp_name, metadata={"format": "pt"})
            target = join_path(path, self.MODEL_FNAME)
            if is_url(path):
                upload(tmp_name, str(target), save_overwrite=True)
            else:
                os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
