import json
import logging
import os
import re
import tempfile
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, Optional, Tuple, Union

import torch
import torch.distributed as dist
from cached_path import cached_path
from torch.distributed.checkpoint.metadata import Metadata

from ..aliases import PathOrStr
from ..config import Config
from ..distributed.checkpoint import (
    async_save_state_dict,
    get_checkpoint_metadata,
    load_state_dict,
    save_state_dict,
)
from ..distributed.utils import (
    barrier,
    broadcast_object,
    get_fs_local_rank,
    get_rank,
    is_distributed,
)
from ..exceptions import OLMoConfigurationError
from ..io import (
    _http_auth_headers,
    clear_directory,
    dir_is_empty,
    file_exists,
    is_url,
    join_path,
    list_directory,
    normalize_path,
    upload,
)
from ..utils import wait_for
from ..version import VERSION
from .train_module import TrainModule

log = logging.getLogger(__name__)


@dataclass
class CheckpointerConfig(Config):
    """
    A configuration class for building :class:`Checkpointer` instances.
    """

    work_dir: Optional[str] = None
    save_overwrite: Optional[bool] = None
    pre_download: bool = False
    save_thread_count: Optional[int] = None
    load_thread_count: Optional[int] = None
    #  save_process_count: Optional[int] = None
    throttle_uploads: bool = False

    def build(self, process_group: Optional[dist.ProcessGroup] = None, **kwargs) -> "Checkpointer":
        kwargs = {**self.as_dict(exclude_none=True, recurse=False), **kwargs}
        work_dir = kwargs.pop("work_dir", None)
        if work_dir is None:
            raise OLMoConfigurationError("'work_dir' must be provided to build a Checkpointer")
        return Checkpointer(work_dir=Path(work_dir), process_group=process_group, **kwargs)


@dataclass
class CheckpointMetadata(Config):
    ephemeral: Optional[bool] = None
    version: str = VERSION


@dataclass
class Checkpointer:
    """
    Trainer checkpointer.
    """

    METADATA_FNAME: ClassVar[str] = ".metadata.json"
    CHECKPOINT_DIR: ClassVar[str] = "step{step}"
    FS_TIMEOUT: ClassVar[float] = 120.0

    work_dir: Path
    save_overwrite: bool = False
    pre_download: bool = False
    process_group: Optional[dist.ProcessGroup] = None
    save_thread_count: Optional[int] = None
    load_thread_count: Optional[int] = None
    #  save_process_count: Optional[int] = None  # TODO: leads to some MP issues, needs more investigating.
    throttle_uploads: bool = False

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        if get_fs_local_rank() == 0:
            self.work_dir.mkdir(exist_ok=True, parents=True)

    def save(
        self,
        dir: PathOrStr,
        train_module: TrainModule,
        train_state: Dict[str, Any],
        ephemeral: bool = False,
    ):
        """
        Save model, optim, and other training state to a local or remote directory.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dir = normalize_path(dir)
        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            self._save_train_state(dir, wd, train_state)

            # Save model and optim state.
            train_module_dir = f"{dir}/model_and_optim" if is_url(dir) else wd / "model_and_optim"
            if hasattr(train_module, "save_state_dict_direct"):
                # Some train modules (e.g. OLMoDDPTrainModule) manage the distributed
                # checkpoint save/load themselves instead of exposing a flat state dict.
                train_module.save_state_dict_direct(  # type: ignore[attr-defined]
                    train_module_dir,
                    process_group=self.process_group,
                    save_overwrite=self.save_overwrite,
                    thread_count=self.save_thread_count,
                    throttle_uploads=self.throttle_uploads,
                )
            else:
                save_state_dict(
                    train_module_dir,
                    train_module.state_dict_to_save(),
                    process_group=self.process_group,
                    thread_count=self.save_thread_count,
                    #  process_count=self.save_process_count,
                    throttle_uploads=self.throttle_uploads,
                    enable_plan_caching=True,
                    # NOTE: we've already checked and cleared the directory at this point so we can skip
                    # the extra synchronization.
                    _skip_prepare=True,
                )

        self._save_metadata(dir, CheckpointMetadata(ephemeral=ephemeral))

    def save_async(
        self,
        dir: PathOrStr,
        train_module: TrainModule,
        train_state: Dict[str, Any],
        ephemeral: bool = False,
    ) -> Future[None]:
        """
        An async version of :meth:`save()`.
        """
        if is_distributed() and self.process_group is None:
            raise OLMoConfigurationError(
                "a checkpointer process group is required for async checkpointing!"
            )
        if hasattr(train_module, "save_state_dict_direct"):
            raise OLMoConfigurationError(
                f"{type(train_module).__name__} does not support async checkpointing. "
                "Set save_async=False for this train module."
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dir = normalize_path(dir)

        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            self._save_train_state(dir, wd, train_state)

        # Save model and optim state.
        train_module_dir = f"{dir}/model_and_optim"
        future = async_save_state_dict(
            train_module_dir,
            train_module.state_dict_to_save(),
            process_group=self.process_group,
            thread_count=self.save_thread_count,
            #  process_count=self.save_process_count,
            throttle_uploads=self.throttle_uploads,
            enable_plan_caching=True,
            # NOTE: we've already checked and cleared the directory at this point so we can skip
            # the extra synchronization.
            _skip_prepare=True,
        )

        def done_callback(fut: Future):
            del fut
            self._save_metadata(dir, CheckpointMetadata(ephemeral=ephemeral))

        # Upload metadata when everything else is done.
        future.add_done_callback(done_callback)

        return future

    def load(
        self,
        dir: PathOrStr,
        train_module: TrainModule,
        *,
        load_trainer_state: Optional[bool] = None,
        load_optim_state: Optional[bool] = None,
        reset_optimizer_states_on_load: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load model, optim, and other training state from a local or remote checkpoint directory
        created via :meth:`save()` or :meth:`save_async()`.

        :param reset_optimizer_states_on_load: Only honored by train modules that implement the
            direct checkpoint API (:meth:`load_state_dict_direct`). When ``True``, optimizer moments
            are discarded on load and only the main params are restored. When left ``None`` it
            defaults, on an actual resume (i.e. only when trainer state is found), to the train
            module's ``reset_optimizer_states_on_resume`` setting.
        """
        dir = normalize_path(dir)

        # Maybe load trainer state.
        trainer_state: Optional[Dict[str, Any]] = None
        if load_trainer_state is not False:
            # Try loading the given rank's state first, then fall back to rank 0 train state if it
            # doesn't exist, which can happen when we're restoring a checkpoint with a different world size.
            for path in (f"{dir}/train/rank{get_rank()}.pt", f"{dir}/train/rank0.pt"):
                try:
                    trainer_state = torch.load(
                        cached_path(path, quiet=True, headers=_http_auth_headers(path)),
                        weights_only=False,
                    )
                    break
                except FileNotFoundError:
                    pass

            if load_trainer_state is True and trainer_state is None:
                raise FileNotFoundError(f"Missing trainer state in checkpoint dir '{dir}'")

        # The resume-specific optimizer reset only applies to an actual resume, so derive it here
        # (after the trainer-state probe) rather than for every non-``False`` load: loading a
        # standalone model/optimizer checkpoint that has no trainer state must keep its moments.
        if reset_optimizer_states_on_load is None and trainer_state is not None:
            reset_optimizer_states_on_load = getattr(
                train_module, "reset_optimizer_states_on_resume", None
            )

        # Load train module state.
        train_module_dir = f"{dir}/model_and_optim"
        metadata: Optional[Metadata] = None
        if get_rank(self.process_group) == 0:
            try:
                metadata = get_checkpoint_metadata(train_module_dir)
            except FileNotFoundError:
                # Try base directory, which could be the case if user is trying to load model weights
                # (possibly with optimizer state), and not an actual train checkpoint.
                if trainer_state is None:
                    metadata = get_checkpoint_metadata(dir)
                    train_module_dir = dir
                else:
                    raise

        train_module_dir = broadcast_object(train_module_dir)
        if metadata is None:
            metadata = get_checkpoint_metadata(train_module_dir)

        if hasattr(train_module, "load_state_dict_direct"):
            # Some train modules (e.g. OLMoDDPTrainModule) manage the distributed
            # checkpoint save/load themselves instead of exposing a flat state dict.
            train_module.load_state_dict_direct(  # type: ignore[attr-defined]
                train_module_dir,
                process_group=self.process_group,
                pre_download=is_url(dir) and self.pre_download,
                work_dir=self.work_dir,
                thread_count=self.load_thread_count,
                load_optim_state=load_optim_state,
                reset_optimizer_states_on_load=reset_optimizer_states_on_load,
            )
        else:
            state_dict = train_module.state_dict_to_load(metadata, optim=load_optim_state)
            load_state_dict(
                train_module_dir,
                state_dict,
                process_group=self.process_group,
                pre_download=is_url(dir) and self.pre_download,
                work_dir=self.work_dir,
                thread_count=self.load_thread_count,
            )
            train_module.load_state_dict(state_dict)

        return trainer_state

    def write_file(self, dir: PathOrStr, fname: str, contents: Union[str, bytes]) -> PathOrStr:
        """
        Write something to a file in a local or remote directory.

        :param dir: The path/URL of the directory to write the file to.
        :param fname: The name of the file to write, relative to ``dir``.
        :param contents: The contents of the file to write.

        :returns: The path/URL of the file.
        """
        dir = normalize_path(dir)
        fname = normalize_path(fname)

        if not is_url(dir):
            Path(dir).mkdir(exist_ok=True, parents=True)

        mode = "wb" if isinstance(contents, bytes) else "wt"
        tmp_file = tempfile.NamedTemporaryFile(
            mode=mode, delete=False, dir=None if is_url(dir) else dir
        )
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(contents)

            # Ensure all data is written to disk.
            tmp_file.flush()
            if hasattr(os, "fdatasync"):  # only available on linux
                os.fdatasync(tmp_file)  # type: ignore
            tmp_file.close()

            target: PathOrStr
            if is_url(dir):
                target = f"{dir}/{fname}"
                upload(tmp_path, target, save_overwrite=self.save_overwrite)
            else:
                target = Path(dir) / fname
                if target.is_file() and not self.save_overwrite:
                    raise FileExistsError(target)
                target.parent.mkdir(exist_ok=True, parents=True)
                tmp_path.replace(target)

            return target
        finally:
            tmp_file.close()
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def checkpoint_dirname(cls, step: int) -> str:
        return cls.CHECKPOINT_DIR.format(step=step)

    @classmethod
    def dir_is_checkpoint(cls, dir: PathOrStr) -> bool:
        """
        Check if a directory is a checkpoint directory.
        """
        dir = normalize_path(dir)
        if file_exists(f"{dir}/.metadata"):  # just model (and maybe optim state), no trainer state
            return True
        paths_to_check = [
            f"{dir}/train/rank0.pt",
            f"{dir}/model_and_optim/.metadata",
            f"{dir}/{cls.METADATA_FNAME}",
        ]
        for path in paths_to_check:
            if not file_exists(path):
                return False
        return True

    @classmethod
    def find_checkpoints(
        cls, dir: PathOrStr, ephemeral: Optional[bool] = None
    ) -> Generator[Tuple[int, str], None, None]:
        """
        Find checkpoints within a directory.
        """
        dir = normalize_path(dir)
        for path in list_directory(dir):
            name = os.path.basename(path)
            if (m := re.match("^" + cls.CHECKPOINT_DIR.format(step=r"(\d+)$"), name)) is not None:
                step = int(m.group(1))

                # Make sure the directory is a valid checkpoint dir.
                if not cls.dir_is_checkpoint(path):
                    continue

                # Filter out based on ephemeral flag.
                if ephemeral is not None:
                    metadata_url = str(join_path(path, cls.METADATA_FNAME))
                    metadata_path = cached_path(
                        metadata_url, quiet=True, headers=_http_auth_headers(metadata_url)
                    )
                    metadata = CheckpointMetadata.from_file(metadata_path)

                    # Assume not ephemeral for backwards compat.
                    if metadata.ephemeral is None:
                        metadata.ephemeral = False

                    if metadata.ephemeral != ephemeral:
                        continue

                yield step, path

    @classmethod
    def contains_checkpoint(cls, dir: PathOrStr) -> bool:
        """
        Check if a directory is a checkpoint directory or contains a child checkpoint directory.
        """
        if cls.dir_is_checkpoint(dir):
            return True

        try:
            next(cls.find_checkpoints(dir))
            return True
        except (StopIteration, FileNotFoundError):
            return False

    @classmethod
    def latest_checkpoint(cls, dir: PathOrStr) -> str:
        """
        Find the latest checkpoint in a directory of checkpoints.

        :raises FileNotFoundError: If no checkpoints are found.
        """
        dir = normalize_path(dir)
        latest_step: Optional[int] = None
        latest_checkpoint: Optional[str] = None
        for step, path in cls.find_checkpoints(dir):
            if latest_step is None or step > latest_step:
                latest_step = step
                latest_checkpoint = path

        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in '{dir}'")
        else:
            return latest_checkpoint

    def _save_train_state(self, dir: PathOrStr, wd: Path, train_state: Dict[str, Any]):
        train_dir = wd / "train"
        # NOTE: if 'dir' is a URL, the 'wd' will be a different temp dir for each rank.
        if is_url(dir) or get_fs_local_rank() == 0:
            train_dir.mkdir(exist_ok=True, parents=True)
        wait_for(
            train_dir.exists,
            description=f"waiting for '{train_dir}' to be created...",
            timeout=self.FS_TIMEOUT,
        )
        torch.save(train_state, train_dir / f"rank{get_rank()}.pt")

    def _save_metadata(self, dir: PathOrStr, metadata: CheckpointMetadata):
        if get_rank() == 0:
            self.write_file(dir, self.METADATA_FNAME, json.dumps(metadata.as_dict(json_safe=True)))

    def _prepare_dir(self, dir: PathOrStr, ensure_exists: bool = True) -> str:
        dir = normalize_path(dir)

        # Make sure checkpoint directory is empty.
        if self.save_overwrite:
            if get_fs_local_rank() == 0:
                clear_directory(dir)
        elif not dir_is_empty(dir):
            raise FileExistsError(dir)

        # NOTE: We need a barrier here in both cases.
        # 1. If 'self.save_overwrite' then we clear the directory, and anytime we clear a directory in
        # preparation to use it we should have a barrier right after, otherwise one rank might get
        # ahead and write something to the directory prematurely, which then gets removed by the call
        # to `clear_directory()`.
        # 2. And otherwise we are checking if the directory is empty and raising an error if it's not,
        # so we need to make sure all ranks are synchronized on that check before they can proceed
        # to write to the directory.
        barrier()

        if ensure_exists and not is_url(dir):
            if get_fs_local_rank() == 0:
                Path(dir).mkdir(exist_ok=True, parents=True)
            # Ensure the dir exists for all ranks before continuing. This might take a second if we're
            # saving to an NFS drive or something like that.
            wait_for(
                Path(dir).exists,
                description=f"waiting on '{dir}' to be created...",
                timeout=self.FS_TIMEOUT,
            )

        return dir

    def _get_tmp_dir(self, dir: PathOrStr) -> Path:
        # Prepare temporary directory.
        tmp_dir: Path
        if is_url(dir):
            tmp_dir = Path(tempfile.mkdtemp(dir=str(self.work_dir)))
        else:
            tmp_dir = Path(dir).with_name(Path(dir).name + "-tmp")
            if get_fs_local_rank() == 0:
                clear_directory(tmp_dir)
                tmp_dir.mkdir(exist_ok=True, parents=True)
            # NOTE: anytime we clear a directory in preparation to use it we should have a barrier
            # right after, otherwise one rank might get ahead and write something to the directory
            # prematurely, which then gets removed by the call to `clear_directory()`.
            barrier()

            # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
            # creating the temp directory from rank 0 might not be immediately
            # realized in the file systems of the other ranks.
            # So we wait here across all ranks until that tmp checkpoint directory is visible.
            wait_for(
                lambda: tmp_dir.exists(),
                "Waiting for checkpoint directory",
                timeout=self.FS_TIMEOUT,
            )

        return tmp_dir

    def _teardown_tmp_dir(self, dir: PathOrStr, tmp_dir: Path):
        if not is_url(dir):
            # NOTE: When dir is not a URL, tmp dir is shared among ranks so we need a barrier before
            # we tear it down to avoid overwriting the work of other ranks.
            barrier()

            # Replace the temporary directory with the actual checkpoint directory.
            if get_fs_local_rank() == 0:
                # Replace temp directory with target checkpoint directory.
                try:
                    tmp_dir.replace(str(dir))
                except FileNotFoundError:
                    # Caught when another (file-system) local rank 0 has already replaced the tmp directory.
                    # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                    # file-systems.
                    if not Path(dir).exists():
                        raise

            # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
            # replacing the temp directory with the final directory from rank 0 might not be immediately
            # realized in the file systems of the other ranks.
            # So we wait here across all ranks until that final checkpoint directory is visible.
            wait_for(
                lambda: Path(dir).exists(),
                f"waiting for checkpoint directory '{dir}' from rank {get_rank()}",
                timeout=self.FS_TIMEOUT,
            )
        else:
            # NOTE: When dir is a URL, each rank will have its own tmp dir so synchronizing with a
            # barrier isn't necessary.

            # Upload files to final location.
            for path in tmp_dir.glob("**/*"):
                if not path.is_file():
                    continue
                upload(
                    path,
                    f"{dir}/{path.relative_to(tmp_dir)}",
                    save_overwrite=self.save_overwrite,
                )

            # Then remove the temp dir.
            clear_directory(tmp_dir)

    @contextmanager
    def _temporary_wd(self, dir: PathOrStr) -> Generator[Path, None, None]:
        # No need to mkdir here since we'll directly replace the temporary directory with
        # this directory below.
        dir = self._prepare_dir(dir, ensure_exists=False)

        tmp_dir = self._get_tmp_dir(dir)

        yield tmp_dir

        self._teardown_tmp_dir(dir, tmp_dir)
