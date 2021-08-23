import weakref
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Tuple, Type

if TYPE_CHECKING:
    from .event import EmitterGroup


def disconnect_events(emitter, listener):
    """Disconnect all events between an emitter group and a listener.

    Parameters
    ----------
    emitter : napari.utils.events.event.EmitterGroup
        Emitter group.
    listener : Object
        Any object that has been connected to.
    """
    for em in emitter.emitters.values():
        for callback in em.callbacks:
            # *callback* may be either a callable object or a tuple
            # (object, attr_name) where object.attr_name will point to a
            # callable object. Note that only a weak reference to ``object``
            # will be kept. If *callback* is a callable object then it is
            # not attached to the listener and does not need to be
            # disconnected
            if isinstance(callback, tuple) and callback[0] is weakref.ref(
                listener
            ):
                em.disconnect(callback)
    emitter.disconnect()


def iter_connections(
    group: 'EmitterGroup', seen=None
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
