"""
Validators are functions that are intended to be called on a hookimpl at the
moment of hookimpl registration.  They do NOT validate whether the hookimpl
obeys the specification signature (that is handled automatically by pluggy),
instead, they offer additional validation for correctly registered hookimpls.

The utility of this will certainly depend on the hookspec: it may only be
useful for hooks that are designed to return quickly, otherwise the time it
costs to eagerly validate the implementation at registration time may outweigh
the benefits.  However, even if validators aren't called immediately after
registration, this API can still be used to declare what a hook implementation
validator function would look like.

For an example, see "validate_get_reader" below.
"""
import os

from pluggy.hooks import HookImpl

from ..utils.misc import TimeLimit


class HookImplementationError(Exception):
    """An exception raised when a plugin's hook implementation is invalid."""


def validates(specname):
    """Returns a decorator that links the decorated validator to ``specname``.
    """

    def decorator(func):
        setattr(func, 'validates', specname)
        return func

    return decorator


@validates('napari_get_reader')
def validate_get_reader(hookimpl: HookImpl):
    """Validates hook implementations for napari_get_reader.

    Provides a fake path to the get_reader hook, and times the response.

    Parameters
    ----------
    hookimpl : HookImpl
        a hook implementation

    Raises
    ------
    HookImplementationError
        If the hook function does not return a value within the timeout
        specified either in the environment variable NAPARI_GET_READER_TIMEOUT,
        or in the DEFAULT_GET_READER_TIMEOUT.

    HookImplementationError
        If the return value is not None and is not a callable.
    """
    # TODO: this should also be a setting
    DEFAULT_GET_READER_TIMEOUT = 0.1
    maxtime = os.environ.get(
        "NAPARI_GET_READER_TIMEOUT", DEFAULT_GET_READER_TIMEOUT
    )
    try:
        with TimeLimit(seconds=maxtime):
            reader = hookimpl.function('')
    except TimeoutError as e:
        msg = (
            "'napari_get_reader' hook implementation from plugin "
            f"'{hookimpl.plugin_name}' took longer than {maxtime}s "
            "to return a value"
        )
        raise HookImplementationError(msg) from e

    if reader is not None and not callable(reader):
        raise HookImplementationError(
            f"'napari_get_reader' hook implementation from plugin "
            f"'{hookimpl.plugin_name}' returned a value "
            "that was not `None` but was not a callable function."
        )
