# NOTE: Translations are deprecated!
#       We leave this as a shim to not break backward compatibility.

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _log(string: str, *a: Any, **k: Any) -> str:
    logger.debug(
        'Translations are no longer supported. Using "trans._" now just '
        'returns the input string.',
        stacklevel=2,
    )
    return string


class _trans_shim:
    """Shim around _log.

    This makes sure that if any method of this class gets called, it returns
    its own input and logs a debug message. The point is to ensure that users
    of `trans._`, `trans._n` etc. will not be disrupted.
    """

    def __getattr__(cls, name: str) -> Callable:
        return _log


trans = _trans_shim()
translator = _trans_shim()
