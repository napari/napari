# NOTE: Translations are deprecated!
#       We leave this as a shim to not break backward compatibility.

import warnings


def _warn(string, *a, **k):
    warnings.warn(
        'Translations are no longer supported. Using "trans._" now just '
        'returns the input string.',
        stacklevel=2,
    )
    return string


class _trans_shim:
    def __getattr__(cls, name):
        return _warn


trans = _trans_shim()
translator = _trans_shim()
