"""
A high-level distributed checkpointing module with a unified API for saving and
loading both local and remote checkpoints.

Features
--------

- Save with one distributed topology, seamlessly load with a different one. For example,
  with FSDP/FSDP2 you can save/load checkpoints with different world sizes or sharding strategies.
- Save/load directly to/from a remote object store like S3 or GCS. When loading from a remote object
  store each rank only downloads the fraction of the data it needs for its local
  (potentially sharded) tensors.

Overview
--------

Use :func:`save_model_and_optim_state()` to write a checkpoint with your model and optimizer's state,
then use :func:`load_model_and_optim_state()` to load the checkpoint in-place.

You can unshard a checkpoint saved this way with :func:`unshard_checkpoint()`.

API Reference
-------------
"""

from __future__ import annotations

import inspect
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from rich.progress import track
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.aliases import PathOrStr
from olmo_core.config import StrEnum
from olmo_core.io import clear_directory, dir_is_empty, is_url, normalize_path
from olmo_core.utils import gc_cuda, get_element_size, log_once, wait_for

from ..utils import barrier, get_fs_local_rank, is_distributed
from .filesystem import RemoteFileSystemReader, RemoteFileSystemWriter

__all__ = [
    "save_state_dict",
    "async_save_state_dict",
    "load_state_dict",
    "save_model_and_optim_state",
    "async_save_model_and_optim_state",
    "load_model_and_optim_state",
    "unshard_checkpoint",
    "load_keys",
    "get_checkpoint_metadata",
    "UnshardStrategy",
    "UnshardStrategyType",
    "swap_param_keys",
    "prune_state_dict",
    "merge_state_dicts",
]

log = logging.getLogger(__name__)


@torch.no_grad()
def save_state_dict(
    dir: PathOrStr,
    state_dict: Dict[str, Any],
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    thread_count: Optional[int] = None,
    process_count: Optional[int] = None,
    throttle_uploads: bool = False,
    enable_plan_caching: bool = False,
    _skip_prepare: bool = False,
):
    """
    Save an arbitrary state dictionary to a distributed format that can loaded again with
    a different distributed topology.

    .. important::
        Please use :func:`save_model_and_optim_state` to save model/optimizer state dicts instead
        unless you know what you're doing.

    :param dir: Path/URL to save to.
    :param state_dict: The state dict to save.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files.
    :param thread_count: Set this to override the number of threads used while writing data.
    :param process_count: Set this to use a process pool instead of a thread pool when possible
        (currently not compatible with ``throttle_uploads``).
    :param throttle_uploads: If this is set to ``True`` and ``dir`` is a URL then only one
        rank from each node will upload data at a time.
    """
    if not _skip_prepare:
        dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    planner = _new_save_planner(enable_plan_caching=enable_plan_caching)
    dist_cp.state_dict_saver.save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_count=process_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def async_save_state_dict(
    dir: PathOrStr,
    state_dict: Dict[str, Any],
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    thread_count: Optional[int] = None,
    process_count: Optional[int] = None,
    throttle_uploads: bool = False,
    enable_plan_caching: bool = False,
    _skip_prepare: bool = False,
) -> Future[None]:
    """
    An async version of :func:`save_state_dict()`.

    This code first de-stages the state dict on the CPU, then writes it in a separate thread.
    """
    if not _skip_prepare:
        dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    planner = _new_save_planner(enable_plan_caching=enable_plan_caching)
    return dist_cp.state_dict_saver.async_save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_count=process_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def load_state_dict(
    dir: PathOrStr,
    state_dict: Dict[str, Any],
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    thread_count: Optional[int] = None,
):
    """
    Load an arbitrary state dict in-place from a checkpoint saved with :func:`save_state_dict()`.

    :param dir: Path/URL to the checkpoint saved via :func:`save_state_dict()`.
    :param state_dict: The state dict to load the state into.
    :param process_group: The process group to use for distributed collectives.
    :param thread_count: Set the number of threads used for certain operations.
    """
    dir = normalize_path(dir)
    reader = RemoteFileSystemReader(
        dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
    )
    dist_cp.state_dict_loader.load(
        state_dict,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
    )


@torch.no_grad()
def save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    flatten_optimizer_state: bool = False,
    thread_count: Optional[int] = None,
    process_count: Optional[int] = None,
    throttle_uploads: bool = False,
    enable_plan_caching: bool = False,
) -> None:
    """
    Save model and optimizer state dictionaries. The model state can be a sharded model, in which
    case this method will correctly handle the optimizer state to ensure it can be loaded again with
    a different distributed topology through :func:`load_model_and_optim_state()`.

    .. seealso::
        - :func:`load_model_and_optim_state()`
        - :func:`async_save_model_and_optim_state()`
        - :func:`unshard_checkpoint()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`load_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    :param dir: Path/URL to save to.
    :param model: The model to save state from.
    :param optim: The optimizer to save state from.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files.
    :param flatten_optimizer_state: Flatten the optimizer state before saving. This should match
        the setting used when loading the state dict and is needed in a distributed setting when
        the params in some param groups may differ between ranks, such as with pipeline parallelism.
    :param thread_count: Set this to override the number of threads used while writing data.
    :param process_count: Set this to use a process pool instead of a thread pool when possible
        (currently not compatible with ``throttle_uploads``).
    :param throttle_uploads: If this is set to ``True`` and ``dir`` is a URL then only one
        rank from each node will upload data at a time.

    :raises FileExistsError: If the checkpoint dir exists and is non-empty unless ``save_overwrite=True``.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    state_dict = _prepare_state_dict(
        model,
        optim=optim,
        process_group=process_group,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    planner = _new_save_planner(enable_plan_caching=enable_plan_caching)
    dist_cp.state_dict_saver.save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_count=process_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def async_save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    flatten_optimizer_state: bool = False,
    thread_count: Optional[int] = None,
    process_count: Optional[int] = None,
    throttle_uploads: bool = False,
    enable_plan_caching: bool = False,
) -> Future[None]:
    """
    An async version of :func:`save_model_and_optim_state()`.

    This code first de-stages the state dict on the CPU, then writes it in a separate thread.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    state_dict = _prepare_state_dict(
        model,
        optim=optim,
        process_group=process_group,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    planner = _new_save_planner(enable_plan_caching=enable_plan_caching)
    return dist_cp.state_dict_saver.async_save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_count=process_count,
            process_group=process_group,
            throttle_uploads=throttle_uploads,
        ),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def load_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    key_mapping: Optional[Dict[str, str]] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    strict: bool = True,
    flatten_optimizer_state: bool = False,
    thread_count: Optional[int] = None,
):
    """
    Load model and optimizer state in-place from a checkpoint saved via :func:`save_model_and_optim_state()`.
    This method is agnostic to the distributed topology in that it can load checkpoints saved with a different
    distributed topology (e.g. FSDP/FSDP2, DDP).

    .. seealso::
        - :func:`save_model_and_optim_state()`
        - :func:`unshard_checkpoint()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`save_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    .. warning::
        Due to the way :mod:`torch.distributed.checkpoint` works, if you have keys in the checkpoint
        dict that are not present in the current state of the model or optimizer, those keys won't
        be loaded.

        For example, if you added a custom field to one of your optimizer's param groups
        before saving the checkpoint, but don't have that field in the param group of the optimizer
        you're loading into, it won't be added.

        This can cause unexpected behavior if you're not careful. In this case the best thing to do
        is to ensure all keys are in present param groups when you initialize the optimizer, before saving
        or loading a checkpoint.

    :param dir: Path/URL to the checkpoint saved via :func:`save_model_and_optim_state()`.
    :param model: The model to load the state into.
    :param optim: The optimizer to load the state into.
    :param process_group: The process group to use for distributed collectives.
    :param key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.
    :param strict: Load keys strictly.
    :param flatten_optimizer_state: Flatten the optimizer state when loading. This should match
        the setting used when saving the state dict and is needed in a distributed setting when
        the params in some param groups may differ between ranks, such as with pipeline parallelism.
    :param thread_count: Set the number of threads used for certain operations.
    """
    dir = normalize_path(dir)
    state_dict = _prepare_state_dict(
        model, optim, process_group=process_group, flatten_optimizer_state=flatten_optimizer_state
    )
    reader = RemoteFileSystemReader(
        dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
    )
    metadata = reader.read_metadata()

    if key_mapping is not None:
        swap_param_keys(state_dict, key_mapping, metadata=metadata)

    dist_cp.state_dict_loader.load(
        state_dict,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
    )

    if key_mapping is not None:
        swap_param_keys(state_dict, key_mapping, reverse=True, quiet=True)

    dist_cp_sd.set_model_state_dict(
        model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=strict)
    )
    gc_cuda()

    if optim is not None:
        dist_cp_sd.set_optimizer_state_dict(
            model,
            optim,
            state_dict["optim"],
            options=dist_cp_sd.StateDictOptions(
                strict=strict, flatten_optimizer_state_dict=flatten_optimizer_state
            ),
        )
        gc_cuda()


class UnshardStrategyType(StrEnum):
    """
    An enumeration of the unsharding strategies that can be used with :func:`unshard_checkpoint`.
    """

    one_file = "one_file"
    """
    Save the unsharded model state into a one file, and optionally the optimizer state into
    another file. The bigger the model, the more memory this requires. For very big models,
    :data:`one_file_per_tensor` will scale better.
    """

    one_file_per_tensor = "one_file_per_tensor"
    """
    Save each unsharded tensor to its own file. Currently this is not compatible with optimizer
    state.
    """

    chunks = "chunks"
    """
    Like :data:`one_file_per_tensor` but multiple tensors and objects may be grouped into the same file
    up to the limit defined by :data:`UnshardStrategy.chunk_size_bytes`.
    """


@dataclass
class UnshardStrategy:
    """
    Unsharding strategy config for :func:`unshard_checkpoint`.
    """

    name: UnshardStrategyType = UnshardStrategyType.one_file
    """
    The strategy type.
    """

    chunk_size_bytes: Optional[int] = None
    """
    The approximate max chunk size (per file size), in bytes, for the :data:`UnshardStrategyType.chunks` strategy.
    """

    def __post_init__(self):
        if self.name == UnshardStrategyType.chunks and self.chunk_size_bytes is None:
            raise ValueError("'chunk_size_bytes' is required for the 'chunks' strategy")
        if self.chunk_size_bytes is not None and self.name != UnshardStrategyType.chunks:
            raise ValueError("'chunk_size_bytes' is only valid for the 'chunks' strategy")

    @classmethod
    def one_file(cls) -> "UnshardStrategy":
        """
        Use the :data:`UnshardStrategy.one_file` strategy.
        """
        return cls(name=UnshardStrategyType.one_file)

    @classmethod
    def one_file_per_tensor(cls) -> "UnshardStrategy":
        """
        Use the :data:`UnshardStrategy.one_file_per_tensor` strategy.
        """
        return cls(name=UnshardStrategyType.one_file_per_tensor)

    @classmethod
    def chunks(cls, chunk_size_in_bytes: int) -> "UnshardStrategy":
        """
        Use the :data:`UnshardStrategy.chunks` strategy.
        """
        return cls(name=UnshardStrategyType.chunks, chunk_size_bytes=chunk_size_in_bytes)


def unshard_checkpoint(
    dir: PathOrStr,
    target_dir: PathOrStr,
    *,
    optim: Optional[bool] = None,
    save_overwrite: bool = False,
    use_safetensors: bool = False,
    unshard_strategy: Optional[UnshardStrategy] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    quiet: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    Convert a checkpoint saved via :func:`save_model_and_optim_state()` into unsharded
    model and optimizer checkpoint files that can be loaded directly with :func:`torch.load()`
    or `safetensors <https://github.com/huggingface/safetensors>`_ if ``use_safetensors=True``.

    .. warning::
        The safetensors format cannot be used to save optimizer state, since optimizer state
        can contain arbitrary Python objects that need to be pickled.
        Therefore ``optim=True`` and ``use_safetensors=True`` is incompatible.

    .. warning::
        This should only be called in a non-distributed context. Otherwise a :class:`RuntimeError` is raised.

    .. seealso::
        :func:`load_keys()` if you only need to load and unshard certain keys in the checkpoint.

    :param dir: The path/URL to the original checkpoint created via :func:`save_model_and_optim_state()`.
    :param target_dir: The directory to save the unsharded model/optimizer checkpoint files to.
        This must be a local directory. URLs are not supported.
    :param optim: Whether to unshard the optimizer state. This defaults to ``True`` as long as
        ``use_safetensors=False``.
    :param save_overwrite: Overwrite any existing files in ``target_dir``.
    :param use_safetensors: Save the unsharded files with :func:`safetensors.torch.save_file()` instead
        of :func:`torch.save()`.
    :param unshard_strategy: The strategy to use. Defaults to :meth:`UnshardStrategy.one_file`.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.
    :param quiet: Do not show progress messages.

    :return: The path to the unsharded model checkpoint and the path to the unsharded
        optimizer checkpoint if ``optim=True``. These paths may represent files or directories
        depending on the ``unshard_strategy``.

    :raises FileExistsError: If the ``target_dir`` is non-empty and ``save_overwrite=False``.
    """
    # Adapted from `torch.distributed.checkpoint.format_utils.dcp_to_torch_save()`.

    if unshard_strategy is None:
        unshard_strategy = UnshardStrategy.one_file()

    if optim is None:
        optim = (not use_safetensors) and (unshard_strategy.name == UnshardStrategyType.one_file)
    elif optim and use_safetensors:
        raise NotImplementedError("`optim=True` is incompatible with `use_safetensors=True`")
    elif optim and unshard_strategy.name != UnshardStrategyType.one_file:
        raise NotImplementedError(
            f"`optim=True` is incompatible with `unshard_strategy={unshard_strategy}`"
        )

    if is_distributed():
        raise RuntimeError("'unshard_checkpoint' cannot be called in a distributed context")

    dir = normalize_path(dir)

    if is_url(target_dir):
        raise ValueError("'target_dir' must be a local directory")
    target_dir = Path(normalize_path(target_dir))
    target_dir.mkdir(exist_ok=True, parents=True)

    ext = "pt" if not use_safetensors else "safetensors"
    metadata = get_checkpoint_metadata(dir)

    def save(state_dict: Dict[str, Any], path: Path):
        if path.is_file() and not save_overwrite:
            raise FileExistsError(
                f"'{path}' already exists, use `save_overwrite=True` to overwrite it"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        if use_safetensors:
            from safetensors.torch import save_file

            save_file(state_dict, path)
        else:
            torch.save(state_dict, path)

    def get_chunks(prefix: str) -> Tuple[Path, List[Tuple[Path, List[str]]]]:
        assert unshard_strategy is not None
        assert isinstance(target_dir, Path)

        if unshard_strategy.name == UnshardStrategyType.one_file:
            path = target_dir / f"{prefix}.{ext}"
            return path, [(path, [prefix])]
        elif unshard_strategy.name == UnshardStrategyType.one_file_per_tensor:
            path = target_dir / prefix
            chunks = []
            for key in metadata.state_dict_metadata.keys():
                if key.startswith(f"{prefix}."):
                    chunks.append((path / f"{key.replace('.', '-')}.{ext}", [key]))
            return path, chunks
        elif unshard_strategy.name == UnshardStrategyType.chunks:
            assert unshard_strategy.chunk_size_bytes is not None
            path = target_dir / prefix
            chunks = []
            current_size = 0
            current_keys: List[str] = []
            for key, meta in metadata.state_dict_metadata.items():
                if key.startswith(f"{prefix}."):
                    if isinstance(meta, TensorStorageMetadata):
                        size = meta.size.numel() * get_element_size(meta.properties.dtype)
                        if current_keys and current_size + size > unshard_strategy.chunk_size_bytes:
                            chunks.append((path / f"chunk-{len(chunks):05d}.{ext}", current_keys))
                            current_size = 0
                            current_keys = []
                        current_size += size
                        current_keys.append(key)
                    else:
                        # This is a pickled Python object, which is probably pretty small,
                        # so we don't worry about recording the size.
                        current_keys.append(key)
            if current_keys:
                chunks.append((path / f"chunk-{len(chunks):05d}.{ext}", current_keys))
            return path, chunks
        else:
            raise NotImplementedError(unshard_strategy.name)

    def unshard_chunk(prefix: str, path: Path, keys: List[str]):
        state_dict: Dict[str, Any] = _load_unsharded_keys(
            dir, keys, pre_download=pre_download, work_dir=work_dir
        )
        if not state_dict:
            raise RuntimeError(f"missing keys '{keys}' in checkpoint")

        save(state_dict[prefix], path)
        del state_dict
        gc_cuda()

    model_path, model_chunks = get_chunks("model")
    for chunk_path, chunk_keys in track(
        model_chunks, description="Unsharding model chunks...", disable=quiet
    ):
        unshard_chunk("model", chunk_path, chunk_keys)

    optim_path: Optional[Path] = None
    if optim:
        optim_path, optim_chunks = get_chunks("optim")
        for chunk_path, chunk_keys in track(
            optim_chunks, description="Unsharding optim chunks...", disable=quiet
        ):
            unshard_chunk("optim", chunk_path, chunk_keys)

    return model_path, optim_path


def load_keys(
    dir: PathOrStr,
    keys: Iterable[str],
    *,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
) -> Generator[Any, None, None]:
    """
    Load specific keys from a checkpoint.

    .. warning::
        This should only be called in a non-distributed context. Otherwise a :class:`RuntimeError` is raised.

    :param dir: The path/URL to the original checkpoint created via :func:`save_model_and_optim_state()`,
        :func:`save_state_dict`, or one of the other functions in this module.
    :param keys: The keys to load.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.

    :returns: The (unsharded) objects from the checkpoint corresponding to the given keys, in the
        same order as the keys.
    """
    if is_distributed():
        raise RuntimeError("'load_keys' cannot be called in a distributed context")

    dir = normalize_path(dir)
    # validate checkpoint.
    get_checkpoint_metadata(dir)

    keys = list(keys)
    state_dict = _load_unsharded_keys(dir, keys, pre_download=pre_download, work_dir=work_dir)
    for key in keys:
        yield _get_key(state_dict, key, pop=True)


def get_checkpoint_metadata(dir: PathOrStr) -> Metadata:
    """
    Load the metadata from a checkpoint.

    :param dir: The path/URL to the checkpoint.
    """
    dir = normalize_path(dir)
    storage_reader = RemoteFileSystemReader(dir)
    return storage_reader.read_metadata()


def _new_save_planner(enable_plan_caching: bool = False) -> DefaultSavePlanner:
    kwargs: Dict[str, Any] = dict(dedup_save_to_lowest_rank=True)
    # The 'enable_plan_caching' option was only added to 'DefaultSavePlanner' in torch 2.7.
    if "enable_plan_caching" in inspect.signature(DefaultSavePlanner.__init__).parameters:
        kwargs["enable_plan_caching"] = enable_plan_caching
    elif enable_plan_caching:
        log_once(
            log,
            "Ignoring 'enable_plan_caching=True' since it's not supported by this version of "
            "torch (requires torch>=2.7)",
            level=logging.WARNING,
        )
    return DefaultSavePlanner(**kwargs)


def _prepare_env_for_save(
    dir: PathOrStr,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
) -> str:
    dir = normalize_path(dir)

    # Prepare checkpoint folder.
    if save_overwrite:
        if get_fs_local_rank(process_group) == 0:
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
    barrier(process_group)

    if not is_url(dir):
        if get_fs_local_rank(process_group) == 0:
            Path(dir).mkdir(exist_ok=True, parents=True)
        # Ensure the dir exists for all ranks before continuing. This might take a second if we're
        # saving to an NFS drive or something like that.
        wait_for(Path(dir).exists, description=f"Waiting on '{dir}' to be created...")

    return dir


def _prepare_state_dict(
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    flatten_optimizer_state: bool = False,
) -> Dict[str, Any]:
    del process_group  # I feel like these torch functions should take a process group argument.
    sd_options = dist_cp_sd.StateDictOptions(
        full_state_dict=False,
        cpu_offload=True,
        flatten_optimizer_state_dict=flatten_optimizer_state,
    )

    state_dict: Dict[str, Any] = {
        "model": dist_cp_sd.get_model_state_dict(model, options=sd_options)
    }
    if optim is not None:
        state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(model, optim, options=sd_options)

    return state_dict


def swap_param_keys(
    state_dict: Dict[str, Any],
    key_mapping: Dict[str, str],
    metadata: Optional[Metadata] = None,
    reverse: bool = False,
    quiet: bool = False,
):
    for current_key, original_key in key_mapping.items():
        if metadata is not None and f"model.{original_key}" not in metadata.state_dict_metadata:
            continue

        if reverse:
            current_key, original_key = original_key, current_key

        if current_key not in state_dict["model"]:
            continue

        if not quiet:
            log.info(f"Mapping current param '{current_key}' to '{original_key}' in checkpoint")

        state_dict["model"][original_key] = state_dict["model"].pop(current_key)

        if "optim" not in state_dict:
            continue

        if "state" in state_dict["optim"]:  # unflattened optim state dict
            state_dict["optim"]["state"][original_key] = state_dict["optim"]["state"].pop(
                current_key
            )
            for group in state_dict["optim"]["param_groups"]:
                if current_key in group["params"]:
                    idx = group["params"].index(current_key)
                    group["params"][idx] = original_key
                    break
        else:  # flattened optim state dict
            for key in list(state_dict["optim"].keys()):
                if key.startswith(f"state.{current_key}."):
                    new_key = key.replace(f"state.{current_key}.", f"state.{original_key}.", 1)
                    state_dict["optim"][new_key] = state_dict["optim"].pop(key)
                elif key.startswith(f"param_groups.{current_key}."):
                    new_key = key.replace(
                        f"param_groups.{current_key}.", f"param_groups.{original_key}.", 1
                    )
                    state_dict["optim"][new_key] = state_dict["optim"].pop(key)


def _load_unsharded_keys(
    dir: PathOrStr,
    keys: List[str],
    *,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
) -> Dict[str, Any]:
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    state_dict: Dict[str, Any] = {}
    _load_state_dict(
        state_dict,
        storage_reader=RemoteFileSystemReader(dir, pre_download=pre_download, work_dir=work_dir),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )
    return state_dict


def _get_key(state_dict: Dict[str, Any], key: str, pop: bool = False) -> Any:
    if key in state_dict:
        if pop:
            return state_dict.pop(key)
        else:
            return state_dict[key]

    if "." not in key:
        raise KeyError(key)

    root, key = key.split(".", 1)
    if root not in state_dict:
        raise KeyError(root)

    return _get_key(state_dict[root], key, pop=pop)


def _set_key(state_dict: Dict[str, Any], key: str, value: Any):
    if "." not in key or all(["." in k for k in state_dict.keys()]):
        state_dict[key] = value
        return

    root, key = key.split(".", 1)
    if root not in state_dict:
        raise KeyError(root)

    return _set_key(state_dict[root], key, value=value)


def _iter_flat_keys(state_dict: Dict[str, Any], prefix: str = "") -> Generator[str, None, None]:
    for key, item in state_dict.items():
        if isinstance(item, dict):
            yield from _iter_flat_keys(item, prefix=key + ".")
        else:
            yield prefix + key


def prune_state_dict(state_dict: Dict[str, Any], allowed_keys: Set[str]) -> Set[str]:
    """
    Prune a state dict by removing all keys not in ``allowed_keys``.

    :returns: The keys that were pruned.
    """
    pruned_keys = set()
    flat_keys = list(_iter_flat_keys(state_dict))
    for key in flat_keys:
        if key not in allowed_keys:
            _get_key(state_dict, key, pop=True)
            pruned_keys.add(key)
    return pruned_keys


def merge_state_dicts(lhs: Dict[str, Any], rhs: Dict[str, Any]):
    """
    Merge ``rhs`` state dict into ``lhs``.
    """
    keys_to_set = set(_iter_flat_keys(rhs)) - set(_iter_flat_keys(lhs))
    for key in keys_to_set:
        _set_key(lhs, key, _get_key(rhs, key))
