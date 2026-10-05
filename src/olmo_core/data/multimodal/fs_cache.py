"""A filesystem cache decorator whose cache directory comes from a configurable env var.

:func:`olmo_core.fs_cache.maybe_cache` always reads ``OLMO_CORE_FS_CACHE_DIR``. The multimodal
data sources cache expensive *verification* results (e.g. file digests) under a directory of their
own, so a run can enable that cache without also caching every other ``maybe_cache`` call.
"""

import functools as ft
import logging
import os
import pickle
import tempfile
import typing
from pathlib import Path
from typing import Callable, Optional, TypeVar

from filelock import FileLock

from olmo_core.fs_cache import _deterministic_hash

__all__ = ["maybe_cache_env"]

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])


def maybe_cache_env(
    cache_dir_env_var: str, *, condition: Optional[Callable[..., bool]] = None
) -> Callable[[F], F]:
    """
    Like :func:`olmo_core.fs_cache.maybe_cache`, but the persistent cache directory is read from
    ``cache_dir_env_var``. Caching is disabled when that variable is unset.

    Arguments must be JSON-serializable. The result must be pickle-able.

    :param cache_dir_env_var: Environment variable naming the cache directory.
    :param condition: Optional predicate deciding whether a call is cached.
    """

    def decorator(user_function: F) -> F:
        @ft.wraps(user_function)
        def wrapper(*args, **kwargs):
            cache_dir = os.environ.get(cache_dir_env_var)
            if cache_dir is None or (condition is not None and not condition(*args, **kwargs)):
                return user_function(*args, **kwargs)

            key = f"{user_function.__qualname__}-{_deterministic_hash((args, kwargs))}"
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            result_path = cache_path / f"{key}.pkl"
            with FileLock(cache_path / f"{key}.lock"):
                if result_path.exists():
                    log.debug("Loading result for %s() from cache", user_function.__qualname__)
                    with result_path.open("rb") as f:
                        return pickle.load(f)
                result = user_function(*args, **kwargs)
                tmp_file = tempfile.NamedTemporaryFile(
                    mode="wb", dir=cache_path, prefix=key, suffix=".tmp", delete=False
                )
                tmp_path = Path(tmp_file.name)
                try:
                    pickle.dump(result, tmp_file)
                    tmp_file.flush()
                    if hasattr(os, "fdatasync"):  # only available on linux
                        os.fdatasync(tmp_file)  # type: ignore
                    tmp_file.close()
                    tmp_path.replace(result_path)
                finally:
                    tmp_file.close()
                    tmp_path.unlink(missing_ok=True)
                return result

        return typing.cast(F, wrapper)

    return decorator
