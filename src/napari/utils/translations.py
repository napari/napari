# NOTE: Translations are deprecated!
#       We leave this as a shim to not break backward compatibility.

import warnings
from collections.abc import Callable
from typing import Any


def _warn(string: str, *a: Any, **k: Any) -> str:
    warnings.warn(
        'Translations are no longer supported. Using "trans._" now just '
        'returns the input string.',
        stacklevel=2,
    )
    return string


class _trans_shim:
    def __getattr__(cls, name: str) -> Callable:
        return _warn


trans = _trans_shim()
translator = _trans_shim()
