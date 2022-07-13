# Napari architecture

The napari codebase can thought to consist of three main components:

* python models describing objects - these are able to operate separately from
  the viewer and do not have any dependencies on user interface classes
    * this code lives in `napari/components` (utility objects) and
     `napari/layers` (objects that contain data)
* qt classes that handle the interactive GUI aspect of the napari viewer
    * the private qt code lives in `napari/_qt` and the smaller public qt
      interface code lives in `napari/qt`
* vispy classes that handles rendering
    * the code for this is private and lives in `napari/_vispy`

The separation of the python models from viewer GUI code allows:

* the python model to be easily run headless without the viewer, for example
  when performing batch analysis
* analysis plugins are able to be developed without worrying about the GUI
  aspect
* napari has the option to move away from the rendering backend currently used
* tests can easily be run headlessly

## Python models and events

Commonly, python models in napari are classes that store information about their
state as an attribute and are the "source of ground truth". When these
attributes are changed an "event" needs to be emitted such that relevant
obserers of the model (such as other classes) can take the appropriate
action. See [An Introduction to the Event Loop in napari](events/event_loop))
for a little more background on events and the event-loop.

One way this is achieved in napari is via getters and setters:

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

Another object can then "listen" for changes in our weather model and register
a callback function with the event emitter of the attribute they would like
to watch:

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

This method is very customizable but requires a lot of boilerplate. The
generic base model `EventedModel` was added to reduce this and
"standardize" this change/emit pattern. The `EventedModel` provides the
following features:

* type validation and coersion on class instantiation and attribute assignment
* event emission after successful attribute assignment

Using `EventedModel` would reduce the above `weather` class code to:

```python
class weather(EventedModel):
    """A simple model to track changes in the weather.

    Parameters
    ----------
    temperature : float
        Degrees in Fahrenheit.
    humidity : float
        Percent humidity
    wind : float
    """
    temperature: float
    humidity: float
    wind: float
```

Currently most of the models in `napari/components/` are `EventedModels` but
not the layer models although there is intention to convert these to
`EventedModels` in the future.

## Qt models

Qt classes are responsible for all napari's user interface elements. There is
generally one to one mapping between Python models and qt models in napari, for
example Python model `Dims` and Qt model `QtDims`.
The Qt class can register callbacks such that when an attribute of the
corresponding Python model changes, the appropriate actions are taken.
The Qt classes are also instantiated with a reference to
the Python model, which gets updated directly when a field is changed via the
GUI.

## Vispy models

Vispy classes are responsible for drawing the canvas contents, thus need to be
informed of any changes to Python models. They achieve this by connecting
callbacks to Python model events just like Qt models.
