from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Optional, Union

from napari.utils.events import EmitterGroup
from napari.utils.events.event import Callback, CallbackRef, EventEmitter

if TYPE_CHECKING:
    from typing import Callable

    from typing_extensions import Protocol

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


def transfer_connections(old_group: EmitterGroup, new_group: EmitterGroup):
    """Transfers connections from an old emitter group to a new one.

    This is useful when an attribute of a type with an EmitterGroup, like an
    EventedModel, is reassigned to a new instance, but you want to maintain the
    behavior triggered by any connections made.

    The existing connections in the old emitter group are not removed, so the
    transfer acts like a copy rather than a move.

    Self-connections are not transferred because it is assumed that the new
    instance will setup the new self-connections itself.

    Parameters
    ----------
    old_group : EmitterGroup
        The emitter group from which to transfer connections.
    new_group : EmitterGroup
        The emitter group to which to transfer connections.
    """
    old_source = old_group.source
    _transfer_callbacks(
        old_emitter=old_group, old_source=old_source, new_emitter=new_group
    )
    for event_name, old_emitter in old_group.emitters.items():
        new_emitter = getattr(new_group, event_name)
        _transfer_callbacks(
            old_emitter=old_emitter,
            old_source=old_source,
            new_emitter=new_emitter,
        )


def _transfer_callbacks(
    *, old_emitter: EventEmitter, old_source: Any, new_emitter: EventEmitter
):
    for cb in old_emitter.callbacks:
        if _get_callback_source(cb) is not old_source:
            new_emitter.connect(cb)


def _get_callback_source(callback: Union[Callback, CallbackRef]) -> Optional:
    if hasattr(callback, 'source'):
        return callback.source
    if isinstance(callback, tuple):
        return callback[0]()
    return None
