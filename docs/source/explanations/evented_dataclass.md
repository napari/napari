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
immediately recognize the main pattern here.  Indeed, the `evented_dataclass` decorator
 wraps the [`dataclasses.dataclass`
decorator](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass)
from the python standard library, adding a few more features relating to events,
and `properties`.  As such, it is *strongly* recommended that you first review the
basics in the [documentation](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass) (or, a more introductory [in-depth tutorial](https://realpython.com/python-data-classes/)).
