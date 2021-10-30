from __future__ import annotations

import weakref
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Tuple, Type

if TYPE_CHECKING:

    from typing_extensions import Protocol

    from .event import EmitterGroup

    class Emitter(Protocol):
        def connect(self, callback: Callable):
            ...

        def disconnect(self, callback: Callable):
            ...


def disconnect_events(emitter, listener):
    """Disconnect all events between an emitter group and a listener.

    Parameters
    ----------
    emitter : napari.utils.events.event.EmitterGroup
        Emitter group.
    listener : Object
        Any object that has been connected to.
    """
    weak_listener = weakref.ref(listener)
    for em in emitter.emitters.values():
        for callback in em.callbacks:
            # *callback* may be either a callable object or a tuple
            # (object, attr_name) where object.attr_name will point to a
            # callable object. Note that only a weak reference to ``object``
            # will be kept. If *callback* is a callable object then it is
            # not attached to the listener and does not need to be
            # disconnected
            if isinstance(callback, tuple) and callback[0] is weak_listener:
                em.disconnect(callback)
    emitter.disconnect()


def iter_connections(
    group: EmitterGroup, seen=None
) -> Iterator[Tuple[Type, Optional[str], Optional[object], str, Callable]]:
    """Yields (SourceType, event_name, receiver, method_name, disconnector)
    for all connections in the EmitterGroup, recursively
    """
    from .event import EmitterGroup

    seen = seen or set()
    for emitter in group.emitters.values():
        for cb in emitter.callbacks:
            if isinstance(cb, EmitterGroup):
                if id(cb) not in seen:
                    seen.add(id(cb))
                    iter_connections(cb, seen)
            elif isinstance(cb, tuple):
                source_type = type(group.source)
                ev_type = emitter.default_args.get("type")
                disconnect = partial(emitter.disconnect, cb)
                yield (source_type, ev_type, cb[0](), cb[1], disconnect)


def connect_setattr(emitter: Emitter, obj, attr: str):
    ref = weakref.ref(obj)

    def _cb(*value):
        setattr(ref(), attr, value[0] if len(value) == 1 else value)

    emitter.connect(_cb)
    # There are scenarios where emitter is deleted before obj.
    # Also there is no option to create weakref to QT Signal
    # but even if keep reference to base object and signal name it is possible to meet
    # problem with C++ "wrapped C/C++ object has been deleted"
    # weakref.finalize(obj, emitter.disconnect, _cb)


def connect_no_arg(emitter: Emitter, obj, attr: str):
    ref = weakref.ref(obj)

    def _cb(*_value):
        getattr(ref(), attr)()

    emitter.connect(_cb)
    # as in connect_setattr
    # weakref.finalize(obj, emitter.disconnect, _cb)


def connect_setattr_value(emitter: Emitter, obj, attr: str):
    """To get value from Event"""
    ref = weakref.ref(obj)

    def _cb(value):
        setattr(ref(), attr, value.value)

    emitter.connect(_cb)
