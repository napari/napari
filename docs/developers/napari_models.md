# Napari models and events

This document explains the links between the three main components of napari:
python models, Qt classes and vispy classes, with code examples. This knowledge
is not necessary to use napari and is more aimed at developers interested in
understanding the inner workings of napari. This document assumes you're
familiar with basic usage of napari.

The three main components:

* python models describing objects - these are able to operate without the GUI
  interface and do not have any dependencies on user interface classes
    * this code lives in `napari/components` (utility objects) and
     `napari/layers` (objects that contain data)
* Qt classes that handle the interactive GUI aspect of the napari viewer
    * the private Qt code lives in `napari/_qt` and the smaller public Qt
      interface code lives in `napari/qt`
* vispy classes that handle rendering
    * the code for this is private and lives in `napari/_vispy`

The separation of the python models from viewer GUI code allows:

* the python model to be easily run headless (without opening the napari GUI
  interface), for example when performing batch analysis



* analysis plugins to be developed without worrying about the GUI
  aspect
* napari to have the option to move away from the rendering backend currently
  used
* tests to be easily run headlessly

## Python models and events

Commonly, python models in napari are classes that store information about their
state as an attribute and are the "source of ground truth". When these
attributes are changed an "event" needs to be emitted such that relevant
observers of the model (such as other classes) can take the appropriate
action.

One way this is achieved in napari is via getters and setters. Let's take
for example the `Dims` class with a selected few attributes:

```python
from napari.utils.events import EmitterGroup

class Dims:
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    ...
    """
    def __init__(self, ndim, ndisplay):
        self._ndim_ = ndim
        self._ndisplay = ndisplay

        # an `EmitterGroup` manages a set of `EventEmitters`
        # we add one emitter for each attribute we'd like to track
        self.events = EmitterGroup(source=self, ndim=None, ndisplay=None)

    # for each attribute, we create a `@property` getter/setter
    # so that we can emit the appropriate event when that attribute
    # is changed using the syntax: ``Dim.attribute = new_value``
    @property
    def ndim(self):
        """Number of dimensions."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = value
        # emit the ndim "changed" event
        self.events.ndim(value=value)

    @property
    def ndisplay(self):
        """Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, value):
        self._ndisplay = value
        # emit the ndisplay "changed" event
        self.events.ndisplay(value=value)
```

Another object can then "listen" for changes in our `Dims` model and register
a callback function with the event emitter of the attribute they would like
to watch:

```python
# create an instance of the model
dims = Dims(ndim=3, ndisplay=2)

# define some callback that should respond to changes in the model
def _update_display(self):
    """
    Updates display for all sliders.
    """
    # the code updating the display code is not relevant for this
    # example thus has been ommited.
    nsteps = self.dims.nsteps
    print(f"Update number of dimensions displayed to {nsteps}")

# register our callback with the model
dims.events.ndisplay.connect(_update_display)

# now, everytime dims.ndisplay is changed, _update_display is called
dim.ndisplay = 3
```

This method is very customizable but requires a lot of boilerplate. The
generic base model `EventedModel` was added to reduce this and
"standardize" this change/emit pattern. The `EventedModel` provides the
following features:

* type validation and coercion on class instantiation and attribute assignment
* event emission after successful attribute assignment

Using `EventedModel` would reduce the above `Dim` class code to:

```python
class Dim(EventedModel):
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    ...
    """
    ndim: float
    ndisplay: float
```

This `Dim` class will automatically emit an event when one of its attributes
changes. Other classes interested in the `Dim` class can register a callback
function that will be executed when an attribute changes.

```python
class DimsDependentClass():
    """A class that needs to 'do something' when Dims attributes change.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Dims object.
    ...

    Attributes
    ----------
    dims : napari.components.dims.Dims
        Dimensions object modeling slicing and displaying.
    ...
    """

    def __init__(self, dims: Dims):
        self.dims = dims
        self.dims.events.ndisplay.connect(self._update_display)
```

Currently most of the models in `napari/components/` are `EventedModels` but
not the layer models although there is intention to convert these to
`EventedModels` in the future.

## Qt classes

Qt classes are responsible for all napari's user interface elements. There is
generally one to one mapping between Python models and Qt models in napari, for
example Python model `Dims` and Qt model `QtDims`.
The Qt class can register callbacks such that when an attribute of the
corresponding Python model changes, the appropriate actions are taken.
The Qt classes are also instantiated with a reference to
the Python model, which gets updated directly when a field is changed via the
GUI.

For example, below is a code snippet showing the `QtDims` class instantiating
with a reference to the python class `Dims` and registering the callback
`_update_display`:

```python
class QtDims(QWidget):
    """Qt view for the napari Dims model.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Dims object to be passed to Qt object.
    ...

    Attributes
    ----------
    dims : napari.components.dims.Dims
        Dimensions object modeling slicing and displaying.
    ...
    """

    def __init__(self, dims: Dims):
        self.dims = dims
        self.dims.events.ndisplay.connect(self._update_display)
```

## Vispy classes

Vispy classes are responsible for drawing the canvas contents, thus need to be
informed of any changes to Python models. They achieve this by connecting
callbacks to Python model events just like Qt models.

For example, below is a code snippet showing the `VispyCamera` class connecting
the function `_on_ndisplay_change`:

```python
class VispyCamera:
    """Vipsy camera for both 2D and 3D rendering.
    """

    def __init__(self, dims: Dims):
        self._dims = dims
        ...

        self._dims.events.ndisplay.connect(
            self._on_ndisplay_change, position='first'
        )
```
