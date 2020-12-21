import dataclasses as _dc
import weakref

from .dataclass import _type_to_compare, is_equal, update_from_dict
from .event import EmitterGroup


def disconnect_events(emitter, listener):
    """Disconnect all events between an emitter group and a listener.

    Parameters
    ----------
    emitter : napari.utils.events.event.EmitterGroup
        Emitter group.
    listener: Object
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


def evented(cls):
    # get field which should be evented
    _fields = [
        _dc._get_field(cls, name, type_)
        for name, type_ in cls.__dict__.get('__annotations__', {}).items()
    ]
    e_fields = {
        fld.name: None
        for fld in _fields
        if fld._field_type is _dc._FIELD and fld.metadata.get("events", True)
    }

    # create an EmitterGroup with an EventEmitter for each field
    if hasattr(cls, 'events') and isinstance(cls.events, EmitterGroup):
        for em in cls.events.emitters:
            e_fields.pop(em, None)
        cls.events.add(**e_fields)
    else:
        cls.events = EmitterGroup(source=cls, auto_connect=False, **e_fields,)

    # create dict with compare functions for fields which cannot be compared
    # using standard equal operator, like numpy arrays.
    # it will be set to __equality_checks__ class parameter.
    compare_dict_base = getattr(cls, "__equality_checks__", {})
    compare_dict = {
        n: t
        for n, t in {
            name: _type_to_compare(type_)
            for name, type_ in cls.__dict__.get('__annotations__', {}).items()
            if name not in compare_dict_base
        }.items()
        if t is not None  # walrus operator is supported from python 3.8
    }
    # use compare functions provided by class creator.
    compare_dict.update(compare_dict_base)

    # modify __setattr__ with version that emits an event when setting
    original_setattr = getattr(cls, '__setattr__')

    def new_setattr(self, name, value):
        set_with_events(self, name, value, original_setattr)

    setattr(cls, '__setattr__', new_setattr)
    setattr(cls, '__equality_checks__', compare_dict)

    setattr(cls, 'asdict', _dc.asdict)
    setattr(cls, 'update', update_from_dict)
    return cls


def set_with_events(self, name, value, original_setattr):
    """Modified __setattr__ method that emits an event when set.

    Events will *only* be emitted if the ``name`` of the attribute being set
    is one of the dataclass fields (i.e. ``name in self.__annotations__``),
    and the dataclass ``__post_init__` method has already been called.

    Also looks for and calls an optional ``_on_name_set()`` method afterwards.

    Order of operations:
        1. Call the original ``__setattr__`` function to set the value
        2. Look for an ``_on_name_set`` method on the object
            a. If present, call it with the current value
            b. That method can do anything (including changing the value, or
               emitting its own events if necessary).  If changing the value,
               it should check to make sure that it is different than the
               current value before setting, or a ``RecursionError`` may occur.
            c. If that method returns ``True``. Return *without* emitting
               an event.
        3. If ``_on_name_set`` has not returned ``True``, then emit an event
           from the EventEmitter with the corresponding ``name`` in the.
           e.g. ``self.events.<name>(value=value)``.

    Parameters
    ----------
    self : C
        An instance of the decorated dataclass of Type[C]
    name : str
        The name of the attribute being set.
    value : Any
        The new value for the attribute.
    fields : set of str
        Only emit events for field names in this set.
    """
    if name not in getattr(self, 'events', {}):
        # fallback to default behavior
        original_setattr(self, name, value)
        return

    # grab current value
    before = getattr(self, name, object())

    # set value using original setter
    original_setattr(self, name, value)

    # if different we emit the event with new value
    after = getattr(self, name)
    if not self.__equality_checks__.get(name, is_equal)(after, before):
        # use gettattr again in case `_on_name_set` has modified it
        getattr(self.events, name)(value=after)  # type: ignore


# Pydandic config so that assigments are validated
class PydanticConfig:
    validate_assignment = True
