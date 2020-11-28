# Using the evented_dataclass decorator

[PR #1475](https://github.com/napari/napari/pull/1475) introduced a custom
`dataclass` decorator (later renamed in
[PR #1958](https://github.com/napari/napari/pull/1958) to `evented_dataclass`)
that simplifies the construction of data models for use in an event-driven
environment like the napari GUI.  This document provides an introduction
to using this dataclass decorator.

## The problem being solved

An extremely common pattern in napari models (such as our `Layer`, `Viewer`, and
`Dims` models) is the construction of a class that has a number of attributes,
each of which should emit an "event" when changed, so that observers of the model
(such as other objects) can take the appropriate action for the given change.
(For a little more background on events and the event-loop, see [An Introduction
to the Event Loop in napari](events/event_loop))
<!-- note: fix that link -->

In code, this might look something like this:

```python
from napari.utils.events import EmitterGroup

class Weather:
    """A simple model to track changes in the weather.

    Parameters
    ----------
    temperature : float
        Degrees in Fahrenheit.
    humidity : float
        Percent humidity
    wind : float
        Wind speed in mph
    """

    def __init__(self, temperature, humidity=70, wind=0):

        self._temperature = temperature
        self._humidity = humidity
        self._wind = wind

        # an `EmitterGroup` manages a set of `EventEmitters`
        # we add one emitter for each attribute we'd like to track
        self.events = EmitterGroup(
            source=self, temperature=None, humidity=None, wind=None
        )

    # for each attribute, we create a `@property` getter/setter
    # so that we can emit the appropriate event when that attribute
    # is changed using the syntax: ``weather.attribute = new_value``

    @property
    def temperature(self):
        """Degrees in Fahrenheit."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        # emit the temperature "changed" event
        self.events.temperature(value=value)

    @property
    def humidity(self):
        """Percent humidity."""
        return self._humidity

    @humidity.setter
    def humidity(self, value):
        self._humidity = value
        # emit the humidity "changed" event
        self.events.humidity(value=value)

    @property
    def wind(self):
        """Wind speed in mph."""
        return self._wind

    @wind.setter
    def wind(self, value):
        self._wind = value
        # emit the wind "changed" event
        self.events.wind(value=value)

```

Now, any *other* object can "listen" for changes in our weather model by
registering a callback function with the event emitter for the attribute they'd
like to watch: for example:

```python
# create an instance of the model
weather = Weather(temperature=72, humidity=65, wind=0)

# define some callback that should respond to changes in the model
def hurricane_watch(event):
    if event.value > 74:
        print("Hurricane! Evacuate!")

# register our callback with the model
weather.events.wind.connect(hurricane_watch)

# now, everytime weather.wind is changed, hurricane_watch is called
weather.wind = 90  # prints: "Hurricane! Evacuate!"

```

**Whew!  That's a lot of boilerplate for a pretty basic model!**

The primary purpose of the `evented_dataclass` is to simplify the construction
of these data-driven models, in effect "standardizing" this change/emit pattern,
reducing boilerplate, and (hopefully) clarifying the model and reducing error.
Before we dig into the details, here's how the `Weather` model above would look
using the napari `evented_dataclass`:


```python
from napari.utils.events import evented_dataclass

@evented_dataclass
class Weather:
    """A simple model to track changes in the weather.

    Parameters
    ----------
    temperature : float
        Degrees in Fahrenheit.
    humidity : float
        Percent humidity
    wind : float
        Wind speed in mph
    """

    temperature: float
    humidity: float = 70
    wind: float = 0
```

## Python `dataclasses.dataclass`

If you're familiar with the builtin python `dataclass` decorator, you will
immediately recognize the main pattern here.  Indeed, the `evented_dataclass`
decorator wraps the [`dataclasses.dataclass`
decorator](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass)
from the python standard library, and all of the standard arguments to
`dataclass` apply, along with a few more features relating to events, and
`properties`.  As such, it is *strongly* recommended that you first review the
basics in the
[documentation](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass)
(or, a more introductory [in-depth
tutorial](https://realpython.com/python-data-classes/)).

Minimally, you should know that the `@dataclass` decorator auto-generates an
`__init__` method by examining the class to find `fields`, where a "field" is
defined as a class variable that has a [type
annotation](https://docs.python.org/3/glossary.html#term-variable-annotation).
So you *must* include a type annotation for your field to be included in the
`__init__` signature.  Variables without a default value are considered
mandatory

```python
In [1]: from dataclasses import dataclass

In [2]: @dataclass
    ...: class Test:
    ...:     x: int  # mandatory parameter
    ...:     y: str = 'hi'  # optional parameter
    ...:     z = 2  # no type annotation, NOT in __init__
    ...:

In [3]: Test?
Init signature: Test(x: int, y: str = 'hi') -> None
Docstring:      Test(x: int, y: str = 'hi')
```

### Handy `dataclass` features

The python documentation for `dataclass` is good, and relatively short, so we
won't duplicated it here, but here are some handy features that may be
particularly useful when defining napari models:

1. Add custom [**post-init
   processing**](https://docs.python.org/3/library/dataclasses.html#post-init-processing)
   by declaring a `__post_init__` method on your class.
2. [**Exclude specific
   variables**](https://docs.python.org/3/library/dataclasses.html#class-variables)
   from dataclass fields by annotating as `typing.ClassVar`.
3. Declare [**init-only
   variables**](https://docs.python.org/3/library/dataclasses.html#init-only-variables)
   (variables that are passed to `__post_init__` but are otherwise *not* counted
   as fields in the dataclass) by annotating as type `dataclasses.InitVar`.
4. Provide a [**`default_factory`
   function**](https://docs.python.org/3/library/dataclasses.html#default-factory-functions)
   to generate the default value for a field using a function (this is important
   for [mutable default
   values](https://docs.python.org/3/library/dataclasses.html#mutable-default-values)
   such as an empty list.)
    ```python
    from dataclasses import dataclass, field

    @dataclass
    class D:
        x: list = field(default_factory=list)
    ```
5. Dataclasses *can* be [used as a base class for
   **inheritance**](https://docs.python.org/3/library/dataclasses.html#inheritance),
   but know that there are some tricky scenarios; specifically: the subclass
   cannot have any non-defaut arguments if the parent class defines any fields
   with defaults. (see [this stack-overflow
   post](https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses)
   for details)

In some cases, you may also need to modify the behavior of the full dataclass, or of individual fields in the dataclass using the `init`, `repr`, `order`, `unsafe_hash`, or `frozen` parameters as well, so it's worth [reviewing their function](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass).

## napari's `evented_dataclass`

The `@evented_dataclass` decorator adds two more parameters to the standard library `dataclass`: `events` and `properties`.  **By default, they are both `True`.**

### `@evented_dataclass(events=True)`

When `events` is `True`, an event will be emitted each time an attribute is modified, as described in the introduction.  For example...

```python
@evented_dataclass(events=True, properties=False)
class A:
    x: int = 1
```

Is *roughly* equivalent to:

```python
class A:
    def __init__(self, x: int = 1):
        self.x = x
        self.events = EmitterGroup(source=self, x=None)

    def __setattr__(self, name, value):
        before = getattr(self, name) # get current value before changing
        object.__setattr__(self, name, value)  # set new value

        # if custom set method `_on_<name>_set` exists, call it
        if custom_setter := getattr(self, f'_on_{name}_set', None):
            # if it returns a truthy value, return without emitting event
            if custom_setter(getattr(self, name)):
                return

        # finally, if the value has changed, emit the appropriate event
        if before != getattr(self, name):
            event_emitter = getattr(self.events, name)  # get emitter
            event_emitter(value=after)  # emit event
```

### `@evented_dataclass(properties=True)`

When `properties` is `True`, attributes are turned in to `property` descriptors that set a private attribute, with the possibility of setting custom getter/setter functions using the special `Property` type.  (In this case, it is possible that the internal/private type differs from the external public value)

```python
from napari.utils.events.dataclass import evented_dataclass, Property

@evented_dataclass(events=False, properties=True)
class A:
    x: int
    y: Property[int, custom_getter, custom_setter] = 2
    # where custom_getter/custom_setter are callables
```

Is *roughly* equivalent to:

```python
class A:
    def __init__(self, x: int, y: int = 2):
        self._x = x
        self._y = y


    @property
    def x(self):
        """Docs for "x" in the class docstring will be added here."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        """Docs for "y" in the class docstring will be added here."""
        return custom_getter(self._y)

    @y.setter
    def y(self, value):
        custom_setter(value)
```

## Examples and Special cases

### Exempting a field from events

Events will *only* be emitted on value change if the name of the attribute being
set is one of the dataclass fields (i.e. ``name in self.__annotations__``), and
the dataclass `__post_init__` method has already been called.

In the following cases, events will *not* be emitted when the value of an
attribute is set
1. If the field is of type
   [`dataclasses.InitVar`](https://docs.python.org/3/library/dataclasses.html#init-only-variables)
2. If the field is of type
   [`typing.ClassVar`](https://docs.python.org/3/library/dataclasses.html#class-variables)
3. If the default value of the field is a call to
   [`dataclasses.field`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field)
   with a `metadata` dict that contains `{'events': False}`

```python
from dataclasses import InitVar, field
from typing import ClassVar

from napari.utils.events.dataclass import evented_dataclass


@evented_dataclass
class Q:
    # ONLY the `a` field will emit events
    a: int = 0
    b: str = field(default="hi", metadata={"events": False})
    c: InitVar[int] = 0
    d: ClassVar[float] = 1.0


q = Q()
print(set(q.events._emitters))  # prints: "{'a'}"
```

Note: event emission will also be skipped if the "new" value is the same as the
existing value

### Getter/setter side-effects & conditional event logic

In some cases, it may be desireable to perform additional tasks or trigger
side-effects upon setting an attribute (in addition to emitting the basic
event). It may also be necessary at times to provide conditional logic that
prevents the emission of an event in specific cases.  For both of these cases,
you may optionally define **`_on_<name>_get`** and/or **`_on_<name>_set`**
methods that will be called when getting/setting the corresponding `<name>`
attribute.


```python
from dataclasses import InitVar, field
from typing import ClassVar

from napari.utils.events.dataclass import evented_dataclass


@evented_dataclass
class E:
    x: int = 1

    def _on_x_get(self, value):
        """Called when e.x is retrieved"""
        # add optional custom logic here
        return value

    def _on_x_set(self, value):
        """Called when `e.x = y` is set"""
        print("setting x =", value)
        if value > 10:
            # return `True` to block event emission
            return True
        # you can override the value if you like,
        # but use the private name to avoid recursion
        self._x = value * 2

# usage:
# in this example, x should double the value then emit an event if val ≤ 10
# otherwise simply set the value without emitting an event.

In [1]: e = E()

In [2]: e.events.x.connect(lambda e: print('x event! x =', e.value))

In [3]: e.x = 3
setting x = 3
x event! x = 6

In [4]: e.x = 15
setting x = 15
```

### Basic type coercion with the `Property` type

Often times an `@property` or `@property.setter` decorator is used if you'd like
to enforce some basic type coercion or type checking when an attribute is
accessed or set.  For this, the `from napari.utils.events.dataclass.Property` is
a special "type" that accepts the "actual" internal type of the attribute, along
with optional getter/setter functions that will be called on the value.

```python
from napari.utils.events.dataclass import evented_dataclass, Property

@evented_dataclass
class F:
    # Property[type, getter_func, setter_func]
    x: Property[int, str, int]

# example usage

In [1]: f = F(1)

# x is returned as str when accessed
In [2]: f.x
Out[2]: '1'

# but the internal/private representation is an int
In [3]: f._x
Out[3]: 1

# type coercion happens during setting
In [4]: f.x = '23'

# so the internal representation is always an int.
In [5]: f._x
Out[5]: 23
```

### Validation using `Property`

The `Property` getter/setters can also be used for validation.  If an exception
is raised by the setter, the value is left unchanged.

```python
def less_than_10(value):
    value = int(value)
    if value >= 10:
        raise ValueError("Value must be less than 10.")
    return value

@evented_dataclass
class F:
    x: Property[int, None, less_than_10] = 5

# example usage

In [1]: f = F()

In [2]: f.x
Out[2]: 5

In [3]: f.x = 13
TypeError: Failed to coerce value 13 in x: ('Value must be less than 10.')

In [4]: f.x  # unchanged
Out[4]: 5
```
