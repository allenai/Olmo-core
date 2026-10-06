"""
Distributed helpers, most of which work in a non-distributed context as well for API unity.
"""

import logging
import math
import os
import socket
from datetime import timedelta
from typing import Callable, List, Optional, TypeVar, Union, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor

from ..exceptions import OLMoEnvironmentError
from ..utils import log_or_print, move_to_device, set_env_var

OLMO_SHARED_FS_ENV_VAR = "OLMO_SHARED_FS"
OLMO_FS_LOCAL_RANK_ENV_VAR = "FS_LOCAL_RANK"
OLMO_LOCAL_RANK_ENV_VAR = "LOCAL_RANK"
OLMO_NUM_NODES_ENV_VAR = "NUM_NODES"
OLMO_LOCAL_WORLD_SIZE_ENV_VAR = "LOCAL_WORLD_SIZE"
BEAKER_HOSTNAME_ENV_VAR = "BEAKER_NODE_HOSTNAME"
TORCH_RANK_ENV_VAR = "RANK"
TORCH_WORLD_SIZE_ENV_VAR = "WORLD_SIZE"
TORCH_MASTER_ADDR_ENV_VAR = "MASTER_ADDR"
TORCH_MASTER_PORT_ENV_VAR = "MASTER_PORT"


log = logging.getLogger(__name__)


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _bootstrap_single_process_distributed_env() -> None:
    """
    Fill in the minimal torchrun-style environment for direct single-process runs.
    """
    missing_rank_env = (
        TORCH_RANK_ENV_VAR not in os.environ and TORCH_WORLD_SIZE_ENV_VAR not in os.environ
    )
    explicit_singleton = (
        os.environ.get(TORCH_WORLD_SIZE_ENV_VAR) == "1"
        and os.environ.get(TORCH_RANK_ENV_VAR, "0") == "0"
    )

    if not missing_rank_env and not explicit_singleton:
        return

    if missing_rank_env:
        log_or_print(
            log,
            "Distributed launch env vars not found; bootstrapping a single-process distributed setup.",
        )

    set_env_var(TORCH_RANK_ENV_VAR, "0")
    set_env_var(TORCH_WORLD_SIZE_ENV_VAR, "1")
    set_env_var(OLMO_LOCAL_RANK_ENV_VAR, "0")
    set_env_var(OLMO_LOCAL_WORLD_SIZE_ENV_VAR, "1")
    set_env_var(OLMO_NUM_NODES_ENV_VAR, "1")
    set_env_var(TORCH_MASTER_ADDR_ENV_VAR, "127.0.0.1")
    if TORCH_MASTER_PORT_ENV_VAR not in os.environ:
        set_env_var(TORCH_MASTER_PORT_ENV_VAR, str(_find_free_local_port()))


def init_distributed(
    backend: str = "nccl",
    timeout: timedelta = timedelta(minutes=30),
    shared_filesytem: Optional[bool] = True,
    **kwargs,
):
    """
    Initialize the distributed process group with the given backend(s) and check/set the
    relevant environment variables.
    This also calls :func:`torch.cuda.set_device()` for backends that support CUDA.

    If the process group is already initialized, this function will skip the initialization
    but still set the environment variables and CUDA device. This allows callers to
    pre-initialize torch.distributed (e.g., without device_id to avoid NCCL hangs when
    multiple process groups exist) before calling this function.
    """
    # To mitigate the memory issue that collectives using async_op=True hold memory longer
    # than they should such as those in tensor parallelism.
    set_env_var("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")

    # Force processes to synchronize at init process group.
    set_env_var("TORCH_DIST_INIT_BARRIER", "1")

    _bootstrap_single_process_distributed_env()

    if shared_filesytem:
        set_env_var(OLMO_SHARED_FS_ENV_VAR, "1")

    # Set host-specific env var defaults.
    if _running_in_beaker():
        node_name = get_node_hostname()
        log_or_print(log, f"Running in beaker on node '{node_name}'")
        multi_node = int(os.environ.get(OLMO_NUM_NODES_ENV_VAR, "1")) > 1

        # See https://beaker-docs.apps.allenai.org/experiments/distributed-training.html
        if "jupiter" in node_name:
            set_env_var("NCCL_IB_HCA", "^=mlx5_bond_0")
            if multi_node:
                # Only for multi-node
                set_env_var("NCCL_SOCKET_IFNAME", "ib")
        elif "pluto" in node_name:
            set_env_var("NCCL_IB_HCA", "^=mlx5_1,mlx5_2")
        elif "augusta" in node_name:
            # NOTE: For single-node training we still need all of these settings and we also
            # need host networking enabled so that the ethernet interface names don't change.
            set_env_var("NCCL_CROSS_NIC", "0")
            #  set_env_var("NCCL_ALGO", "Ring,Tree")
            set_env_var("NCCL_PROTO", "Simple,LL128")
            set_env_var("NCCL_MIN_NCHANNELS", "4")
            set_env_var("NCCL_P2P_NET_CHUNKSIZE", "524288")
            set_env_var("NCCL_P2P_PCI_CHUNKSIZE", "524288")
            set_env_var("NCCL_P2P_NVL_CHUNKSIZE", "1048576")
            set_env_var("NCCL_NVLSTREE_MAX_CHUNKSIZE", "131072")
            set_env_var("NCCL_FASTRAK_NUM_FLOWS", "2")
            set_env_var("NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL", "0")
            set_env_var("NCCL_BUFFSIZE", "8388608")
            set_env_var("NCCL_FASTRAK_USE_SNAP", "1")
            set_env_var("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
            set_env_var("NCCL_NET_GDR_LEVEL", "PIX")
            set_env_var("NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING", "0")
            set_env_var(
                "NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS", str(int(timeout.total_seconds() * 1000))
            )
            #  set_env_var("NCCL_NVLS_ENABLE", "0")
            set_env_var("NCCL_USE_SNAP", "1")
            set_env_var("NCCL_FASTRAK_USE_LLCM", "1")
            set_env_var("NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY", "/dev/aperture_devices")
            # NOTE: This path var must be set prior to launching Python
            #  set_env_var(
            #      "LD_LIBRARY_PATH",
            #      "/var/lib/tcpxo/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
            #      override=True,
            #  )
            set_env_var("NCCL_TUNER_PLUGIN", "libnccl-tuner.so")
            set_env_var(
                "NCCL_TUNER_CONFIG_PATH", "/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto"
            )
            set_env_var(
                "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE",
                "/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto",
            )
            set_env_var("NCCL_FASTRAK_CTRL_DEV", "enp0s12")
            set_env_var(
                "NCCL_FASTRAK_IFNAME",
                "enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
            )
            set_env_var("NCCL_SOCKET_IFNAME", "enp0s12")
            set_env_var(  # Add COLL here to log all collective operations. Extremely verbose, don't use for production.
                "NCCL_DEBUG_SUBSYS", "INIT,NET"
            )

    if backend_supports_cuda(backend):
        # Set CUDA device.
        # NOTE: important to do this *before* initializing the process group to avoid
        # other ranks initializing CUDA on GPU 0.
        device = torch.device(f"cuda:{int(os.environ[OLMO_LOCAL_RANK_ENV_VAR])}")
        torch.cuda.set_device(device)

    if dist.is_initialized():
        existing_backend = dist.get_backend()
        if existing_backend != backend:
            raise OLMoEnvironmentError(
                f"Process group already initialized with backend '{existing_backend}', "
                f"but init_distributed() was called with backend='{backend}'. "
                f"Either use the same backend or do not pre-initialize the process group."
            )
        log_or_print(log, "Process group already initialized, skipping init_process_group")
    else:
        log_or_print(log, f"Initializing process group with {timeout=}...")
        dist.init_process_group(backend, timeout=timeout, **kwargs)

    validate_env_vars()

    log_or_print(
        log,
        f"Global rank {get_rank()} "
        f"= local rank {get_local_rank()} "
        f"= file system local rank {get_fs_local_rank()}",
    )


def validate_env_vars():
    """
    Validate distributed environment variables. This is called internally by :func:`init_distributed()`.
    """
    if not is_distributed():
        return

    if OLMO_LOCAL_RANK_ENV_VAR not in os.environ:
        raise OLMoEnvironmentError(f"Missing env var '{OLMO_LOCAL_RANK_ENV_VAR}'")

    if os.environ.get(OLMO_SHARED_FS_ENV_VAR) != "1" and (
        os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) is None
        and os.environ.get(OLMO_LOCAL_RANK_ENV_VAR) is None
    ):
        raise OLMoEnvironmentError(
            f"Missing env var '{OLMO_FS_LOCAL_RANK_ENV_VAR}'/'{OLMO_LOCAL_RANK_ENV_VAR}' for non-shared filesystem. "
            f"If this is a shared filesystem you can set '{OLMO_SHARED_FS_ENV_VAR}=1' instead."
        )

    if OLMO_NUM_NODES_ENV_VAR not in os.environ and OLMO_LOCAL_WORLD_SIZE_ENV_VAR not in os.environ:
        raise OLMoEnvironmentError(
            f"Missing either '{OLMO_NUM_NODES_ENV_VAR}' or '{OLMO_LOCAL_WORLD_SIZE_ENV_VAR}' env vars"
        )


def _running_in_beaker() -> bool:
    return BEAKER_HOSTNAME_ENV_VAR in os.environ


def get_node_hostname() -> str:
    """
    Get the hostname of the node.
    """
    if BEAKER_HOSTNAME_ENV_VAR in os.environ:
        return os.environ[BEAKER_HOSTNAME_ENV_VAR]
    else:
        import socket

        return socket.gethostname()


def is_distributed() -> bool:
    """
    Check if in a distributed context.
    """
    return dist.is_available() and dist.is_initialized()


def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Wait for all ranks in the group.
    """
    if is_distributed():
        dist.barrier(group)


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the rank within the process group.
    """
    if is_distributed():
        return dist.get_rank(group)
    else:
        return 0


def get_global_rank(group_rank: int, group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Translate a rank within a group into it's global rank.
    """
    if group is None or not is_distributed():
        return group_rank
    else:
        return dist.get_global_rank(group, group_rank)


def get_local_rank() -> int:
    """
    Get the local rank within the current node.

    .. warning::
        This relies on the environment variable ``LOCAL_RANK`` being set correctly, but
        will always return 0 if a distributed process group has not been initialized.

    :returns: The rank.
    """
    if is_distributed():
        return int(os.environ.get(OLMO_LOCAL_RANK_ENV_VAR) or 0)
    else:
        return 0


def get_fs_local_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then :func:`get_fs_local_rank()` will be equivalent
    to :func:`get_rank()`, but if nodes do not share the same filesystem then
    :func:`get_fs_local_rank()` will be equivalent to :func:`get_local_rank()`.

    .. warning::
        This relies on some environment variables to determine the correct rank.
        If you are using a shared filesystem across nodes, you can simply set the environment
        variable ``OLMO_SHARED_FS=1``. Otherwise you can set ``FS_LOCAL_RANK`` for each process.

        This will always return 0 if a distributed process group has not been initialized.

    :returns: The rank.
    """
    if not is_distributed():
        return 0
    elif os.environ.get(OLMO_SHARED_FS_ENV_VAR) == "1":
        return int(os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) or get_rank(group))
    else:
        return int(os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) or get_local_rank())


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the world size of the default distributed process group.

    .. warning::
        This will always return 1 if a distributed group has not been initialized.
    """
    if is_distributed():
        return dist.get_world_size(group)
    else:
        return 1


def get_local_world_size() -> int:
    """
    Get the local world size within the default distributed process group.

    .. warning::
        This relies on the 'LOCAL_WORLD_SIZE' env var but will always return 1 if a distributed
        process group has not been initialized.

    :returns: The local world size.
    """
    if not is_distributed():
        return 1
    else:
        return int(os.environ[OLMO_LOCAL_WORLD_SIZE_ENV_VAR])


def get_num_nodes() -> int:
    """
    Get the number of nodes in the default distributed process group.

    .. warning::
        This relies on either the 'NUM_NODES' or 'LOCAL_WORLD_SIZE' env var, but will always
        return 1 if a distributed process group has not been initialized.

    :returns: The number of nodes.
    """
    if not is_distributed():
        return 1
    elif OLMO_NUM_NODES_ENV_VAR in os.environ:
        return int(os.environ[OLMO_NUM_NODES_ENV_VAR])
    else:
        return get_world_size() // get_local_world_size()


V = TypeVar("V", bool, int, float, torch.Tensor)


def synchronize_value(
    value: V, device: torch.device, src: int = 0, group: Optional[dist.ProcessGroup] = None
) -> V:
    """
    Synchronize a value across the distributed process group.
    """
    if dist.is_available() and dist.is_initialized():
        is_tensor = isinstance(value, torch.Tensor)
        value_tensor = (
            move_to_device(value, device)
            if is_tensor
            else move_to_device(torch.tensor(value), device)
        )  # type: ignore
        dist.broadcast(value_tensor, src, group=group)
        return value_tensor if is_tensor else value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(
    flag: bool, device: torch.device, group: Optional[dist.ProcessGroup] = None
) -> bool:
    """
    Synchronize a boolean across the distributed process group.
    """
    return synchronize_value(flag, device, group=group)


def all_reduce_value(
    value: V, device: torch.device, op=dist.ReduceOp.SUM, group: Optional[dist.ProcessGroup] = None
) -> V:
    """
    All reduce a value across the distributed process group.
    """
    if dist.is_available() and dist.is_initialized():
        is_tensor = isinstance(value, torch.Tensor)
        value_tensor = (
            move_to_device(value, device)
            if is_tensor
            else move_to_device(torch.tensor(value), device)
        )  # type: ignore
        dist.all_reduce(value_tensor, op=op, group=group)
        return value_tensor if is_tensor else value_tensor.item()  # type: ignore
    else:
        return value


T = TypeVar("T")


def broadcast_object(obj: T, src: int = 0, group: Optional[dist.ProcessGroup] = None) -> T:
    """
    Broadcast an object using pickle to all ranks in the process group.
    """
    if not is_distributed():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src, group=group)
    return object_list[0]


def all_gather(
    tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None
) -> List[torch.Tensor]:
    """
    All-gather tensors from the whole group into a list.
    """
    if not is_distributed():
        return [tensor]

    shapes = all_gather_object(tensor.shape, group=group)
    output_list = [
        move_to_device(torch.zeros(shape, dtype=tensor.dtype), tensor.device) for shape in shapes
    ]
    dist.all_gather(output_list, tensor, group=group)
    return output_list


def all_gather_object(obj: T, group: Optional[dist.ProcessGroup] = None) -> List[T]:
    """
    All-gather an object using pickle to all ranks in a process group.
    """
    if not is_distributed():
        return [obj]

    output_list = [obj] * get_world_size(group)
    dist.all_gather_object(output_list, obj, group=group)
    return output_list


def get_mesh_coordinates(mesh: "DeviceMesh", rank: Optional[int] = None) -> Optional[List[int]]:
    """
    Calculate the coordinates of a global rank on a device mesh.

    :param mesh: The device mesh.
    :param rank: The global rank. If ``None``, the current global rank is used.

    :return: The coordinates or ``None`` if the rank is not part of the mesh.
    """
    rank = rank if rank is not None else get_rank()
    rank_coords = (mesh.mesh == rank).nonzero()
    assert rank_coords.size(0) in (0, 1)
    return rank_coords[0].tolist() if rank_coords.size(0) > 0 else None


def backend_supports_cuda(backend: Optional[str] = None) -> bool:
    """
    Check if a distributed backend supports CUDA tensors.
    """
    if backend is None and not is_distributed():
        return torch.cuda.is_available()

    backend = backend or dist.get_backend()
    if "nccl" in backend:
        return True
    else:
        return False


def backend_supports_cpu(backend: Optional[str] = None) -> bool:
    """
    Check if a distributed backend supports CPU tensors.
    """
    if backend is None and not is_distributed():
        return True

    backend = backend or dist.get_backend()
    if "gloo" in backend or "mpi" in backend:
        return True
    else:
        return False


def get_reduce_divide_factor(world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
        factor *= 2
    return float(factor)


def get_local_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        x = x.to_local()
        # An `AsyncCollectiveTensor` might be returned, which means the local tensor is not ready
        # yet (i.e. communication is not finished). In this case we need to call `.wait()`
        # to wait the local tensor to be ready.
        if hasattr(x, "wait"):
            return x.wait()  # type: ignore
        else:
            return x
    else:
        return x


def get_full_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        return x.full_tensor()
    else:
        return x


def distribute_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not isinstance(source, DTensor):
        return get_full_tensor(target)

    if isinstance(target, DTensor):
        if target.device_mesh == source.device_mesh and target.placements == source.placements:
            return target
        else:
            return target.redistribute(device_mesh=source.device_mesh, placements=source.placements)

    return distribute_tensor(target, device_mesh=source.device_mesh, placements=source.placements)


def do_n_at_a_time(
    f: Callable[[], T],
    *,
    n: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
) -> T:
    """
    Call a function ``f`` in a distributed context from at most ``n`` ranks at a time.

    All ranks will eventually call the given function exactly once, at which point this function
    will return.

    :param f: The function to call from each rank.
    :param n: The level of concurrency, i.e. how many ranks are allowed to call ``f`` at once.
        This defaults to the number of nodes, in which case one rank from each node will
        call ``f`` at a time.
    :param process_group: The process group to use.
    """
    world_size = world_size if world_size is not None else get_world_size(process_group)
    local_rank = local_rank if local_rank is not None else get_rank(process_group)
    n = n if n is not None else get_num_nodes()
    group_count = math.ceil(world_size / n)
    group_rank = local_rank % group_count
    result: Optional[T] = None
    for active_group in range(group_count):
        if group_rank == active_group:
            result = f()
        barrier(process_group)
    return cast(T, result)


class _HiddenTensor:
    def __init__(self, x: torch.Tensor):
        self.x = x

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(self, *args, **kwargs) -> "_HiddenTensor":
        return _HiddenTensor(self.x.to(*args, **kwargs))


def hide_from_torch(x: torch.Tensor) -> _HiddenTensor:
    return _HiddenTensor(x)


def unhide_from_torch(x: Union[torch.Tensor, _HiddenTensor]) -> torch.Tensor:
    if isinstance(x, _HiddenTensor):
        return x.x
    else:
        return x
