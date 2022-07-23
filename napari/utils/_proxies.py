import re
import sys
import warnings
from typing import Any, Callable, Generic, TypeVar, Union

import wrapt

from ..utils import misc
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
    """Proxy to prevent private attribute and item access, recursively."""

    __wrapped__: _T

    def __getattr__(self, name: str):
        if name.startswith("_") and not (
            name.startswith('__') and name.endswith('__')
        ):
            # allow napari to access private attributes and get an non-proxy
            frame = sys._getframe(1) if hasattr(sys, "_getframe") else None
            if frame.f_code.co_filename.startswith(misc.ROOT_DIR):
                return super().__getattr__(name)

            typ = type(self.__wrapped__).__name__
            warnings.warn(
                trans._(
                    "Private attribute access ('{typ}.{name}') in this context (e.g. inside a plugin widget or dock widget) is deprecated and will be unavailable in version 0.5.0",
                    deferred=True,
                    name=name,
                    typ=typ,
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            # name = f'{type(self.__wrapped__).__name__}.{name}'
            # raise AttributeError(
            #     trans._(
            #         "Private attribute access ('{typ}.{name}') not allowed in this context.",
            #         deferred=True,
            #         name=name,
            #         typ=typ,
            #     )
            # )
        return self.create(super().__getattr__(name))

    def __getitem__(self, key):
        return self.create(super().__getitem__(key))

    def __repr__(self):
        return repr(self.__wrapped__)

    def __dir__(self):
        return [x for x in dir(self.__wrapped__) if not _SUNDER.match(x)]

    @classmethod
    def create(cls, obj: Any) -> Union['PublicOnlyProxy', Any]:
        # restrict the scope of this proxy to napari objects
        mod = getattr(type(obj), '__module__', None) or ''
        if not mod.startswith('napari'):
            return obj
        if isinstance(obj, PublicOnlyProxy):
            return obj  # don't double-wrap
        if callable(obj):
            return CallablePublicOnlyProxy(obj)
        return PublicOnlyProxy(obj)


class CallablePublicOnlyProxy(PublicOnlyProxy[Callable]):
    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)
