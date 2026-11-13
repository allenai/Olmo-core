from typing import TypeVar

T = TypeVar("T")


def beta_feature(f: T) -> T:
    """
    Mark a class or function as a beta feature.
    """
    if f.__doc__ is None:
        f.__doc__ = ""

    f.__doc__ += """

    .. warning::
        This is a beta feature! The API is subject to change even with minor and patch releases.
        If you choose to use this feature please read the `CHANGELOG <https://github.com/allenai/Olmo-core/blob/main/CHANGELOG.md>`_
        before upgrading your version of this library.

    """

    return f
