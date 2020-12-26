import numpy as np
from pydantic import BaseModel, Extra

from .events.dataclass import _type_to_compare, is_equal
from .events.event import EmitterGroup


class _ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (Array,), {'__dtype__': t})


class Array(np.ndarray, metaclass=_ArrayMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, '__dtype__', None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = tuple()

        result = np.array(val, dtype=dtype, copy=False, ndmin=len(shape))
        if shape and len(shape) != len(result.shape):  # ndmin guarantees this
            raise ValueError('Shape mismatch')

        if any(
            (shape[i] != -1 and shape[i] != result.shape[i])
            for i in range(len(shape))
        ):
            result = result.reshape(shape)
        return result


def evented_model(cls):
    """Create an evented model.

    An event emitter is added to the model, and every field is
    modified such that it emits an event with payload `value`
    when it is set to a new value.

    Parameters
    ----------
    cls : pydantic.BaseModel
        Model to make evented.

    Returns
    -------
    pydantic.BaseModel
        Evented model.
    """

    _fields = list(cls.__fields__)
    e_fields = {fld: None for fld in _fields}

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


JSON_ENCODERS = {np.ndarray: lambda arr: arr.tolist()}


class ConfiguredModel(BaseModel):
    # Pydandic config so that assigments are validated
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        underscore_attrs_are_private = True
        use_enum_values = True
        validate_all = True
        extra = Extra.forbid
        json_encoders = JSON_ENCODERS
