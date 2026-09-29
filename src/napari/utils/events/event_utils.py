from __future__ import annotations

import contextlib
import inspect
import logging
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from psygnal import SignalGroup

from napari.utils.events.event import EmitterGroup

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    class Emitter(Protocol):
        def connect(self, callback: Callable, /) -> object: ...

        def disconnect(self, callback: Callable, /) -> object: ...


_logger = logging.getLogger(__name__)


def _get_methods(obj: Any) -> list[Callable]:
    """Get all the (bound) instance methods of an object."""
    methods = []

    for cls in obj.__class__.__mro__:
        for name, value in cls.__dict__.items():
            if name in methods:
                continue

            if inspect.isfunction(value):
                methods.append(getattr(obj, name))

    return methods


def disconnect_events(
    emitter: EmitterGroup | SignalGroup, listener: object
) -> None:
    """Disconnect all events between an emitter group and a listener.

    Parameters
    ----------
    emitter : napari.utils.events.event.EmitterGroup
        Emitter group.
    listener : Object
        Any object that has been connected to.
    """
    if isinstance(emitter, EmitterGroup):
        emitter.disconnect(listener)
        for em in emitter.emitters.values():
            em.disconnect(listener)
    elif isinstance(emitter, SignalGroup):
        if callable(listener):
            emitter.disconnect(listener)
        else:
            # TODO: this currently is not supported in psygnal; one needs to
            # manually disconnect from each method
            for method in _get_methods(listener):
                with contextlib.suppress(TypeError):
                    # suppressing type error because it may happen when the method
                    # requires extra arguments due to how psygnal deals with
                    # weak references (we don't care, it can't be connected so
                    # there's nothing to disconnect)
                    emitter.disconnect(method)


@runtime_checkable
class _EventedObjectProtocol(Protocol):
    @property
    def events(self) -> EmitterGroup | SignalGroup: ...


@runtime_checkable
class _EventedModelProtocol(_EventedObjectProtocol, Protocol):
    @property
    def model_fields(self) -> Iterable: ...


@runtime_checkable
class _EventedMappingProtocol(_EventedObjectProtocol, Protocol):
    def values(self) -> Iterable: ...


@runtime_checkable
class _EventedContainerProtocol(_EventedObjectProtocol, Protocol):
    def __iter__(self) -> Iterator: ...


def _disconnect_all_events(
    evented_object: _EventedObjectProtocol, listener: object
) -> None:
    disconnect_events(evented_object.events, listener)

    values: Iterable = []
    if isinstance(evented_object, _EventedModelProtocol):
        values = [
            getattr(evented_object, name)
            for name in evented_object.__class__.model_fields
        ]
    elif isinstance(evented_object, _EventedMappingProtocol):
        values = evented_object.values()
    elif isinstance(evented_object, _EventedContainerProtocol):
        values = evented_object
    else:
        # fallback (e.g: for non-evented model object like Layer) which
        # just looks for things called like the emitter name
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', (DeprecationWarning, FutureWarning)
            )
            # because we just grab attributes, they might be deprecated stuff, no need
            # to show these warnings
            if isinstance(evented_object.events, EmitterGroup):
                values = [
                    attr
                    for emitter_name in evented_object.events.emitters
                    if (attr := getattr(evented_object, emitter_name, None))
                    is not None
                ]
            elif isinstance(evented_object.events, SignalGroup):
                values = [
                    attr
                    for signal_name in evented_object.events.signals
                    if (attr := getattr(evented_object, signal_name, None))
                    is not None
                ]
    for value in values:
        if isinstance(value, _EventedObjectProtocol):
            _disconnect_all_events(value, listener)


def connect_setattr(
    emitter: Emitter,
    obj,
    attr: str,
    convert_fun: Callable[[Any], Any] | None = None,
) -> None:
    ref = weakref.ref(obj)
    if convert_fun:
        # Handle passed `convert_func` function to map emitted values to valid
        # values accepted for the receiver object attribute.
        # A `convert_func` is needed to, for example, map `Qt.CheckState`
        # values to boolean ones when a `QCheckBox` value change is connected
        # to a layer attribute.
        # See napari/napari#8154
        def _cb(*value):
            if (ob := ref()) is None:
                emitter.disconnect(_cb)
                return

            value = tuple(convert_fun(x) for x in value)  # pyrefly: ignore [not-callable]
            setattr(ob, attr, value[0] if len(value) == 1 else value)
    else:

        def _cb(*value):
            if (ob := ref()) is None:
                emitter.disconnect(_cb)
                return

            setattr(ob, attr, value[0] if len(value) == 1 else value)

    emitter.connect(_cb)
    # There are scenarios where emitter is deleted before obj.
    # Also there is no option to create weakref to QT Signal
    # but even if keep reference to base object and signal name it is possible to meet
    # problem with C++ "wrapped C/C++ object has been deleted"

    # In all of these 3 functions, this should be uncommented instead of using
    # the if clause in _cb but that causes a segmentation fault in tests
    # weakref.finalize(obj, emitter.disconnect, _cb)


def connect_no_arg(emitter: Emitter, obj, attr: str):
    ref = weakref.ref(obj)

    def _cb(*_value):
        if (ob := ref()) is None:
            emitter.disconnect(_cb)
            return
        getattr(ob, attr)()

    emitter.connect(_cb)
    # as in connect_setattr
    # weakref.finalize(obj, emitter.disconnect, _cb)


def connect_setattr_value(emitter: Emitter, obj, attr: str):
    """To get value from Event"""
    ref = weakref.ref(obj)

    def _cb(value):
        if (ob := ref()) is None:
            emitter.disconnect(_cb)
            return
        setattr(ob, attr, value.value)

    emitter.connect(_cb)
    # weakref.finalize(obj, emitter.disconnect, _cb)
