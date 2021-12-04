import re
from typing import Generic, TypeVar

import wrapt

from ..utils.translations import trans

_T = TypeVar("_T")


class ReadOnlyWrapper(wrapt.ObjectProxy):
    """
    Disable item and attribute setting with the exception of  ``__wrapped__``.
    """

    def __setattr__(self, name, val):
        if name != '__wrapped__':
            raise TypeError(
                trans._(
                    'cannot set attribute {name}',
                    deferred=True,
                    name=name,
                )
            )

        super().__setattr__(name, val)

    def __setitem__(self, name, val):
        raise TypeError(
            trans._('cannot set item {name}', deferred=True, name=name)
        )


_SUNDER = re.compile('^_[^_]')


class PublicOnlyProxy(wrapt.ObjectProxy, Generic[_T]):
    """Prevent's private access."""

    __wrapped__: _T

    def __getattr__(self, name: str):
        if name.startswith('_'):
            name = f'{type(self.__wrapped__).__name__}.{name}'
            raise AttributeError(
                trans._(
                    "Private attribute access ('{name}') not allowed in this context.",
                    deferred=True,
                    name=name,
                )
            )
        attr = super().__getattr__(name)
        return (
            CallablePublicOnlyProxy(attr)
            if callable(attr)
            else PublicOnlyProxy(attr)
        )

    def __repr__(self):
        return repr(self.__wrapped__)

    def __dir__(self):
        return [x for x in dir(self.__wrapped__) if not _SUNDER.match(x)]


class CallablePublicOnlyProxy(PublicOnlyProxy[_T]):
    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)
