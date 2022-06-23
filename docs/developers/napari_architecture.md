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

## EventedModel

Users are able to interact with the napari viewer via both the python console
and the GUI interface. This means that python models and qt objects
need to communicate with each other. This is often achieved via the generic
model base class `EventedModel`. This class inherits from pydantic `BaseModel`,
provides type checking and coercion for fields and will emit events when
fields change. There is usually one to one mapping between core python
models and qt classes. These qt classes are instantiated with a reference to
the python model, which gets updated directly when a field is changed via the
GUI.

A simple example of this in napari code is `Dims` class, looking specically at
the `current_step` field, which specifies the slider position for each dims
slider. Below is the napari code for this field, keeping only the code relevant
for the example:

```python
class Dims(EventedModel):
    current_step: Tuple[int, ...] = ()
```

The `Dims` class inherits from the `EventedModel` and has the field
`current_step` which is a tuple if ints.

The matching qt class is `QtDims` which is instantiated with a reference to
the `Dims` object, allowing direct updates if the field is changed via the GUI.
We also connect the method `self._update_slider` such that it is called
when the field `current_step` changes:

```python
class QtDims(QWidget):

    def __init__(self, dims: Dims, parent=None):

      self.dims.events.current_step.connect(self._update_slider)
```

Note napari layer models are not `EventedModel`s yet but there is intention
to convert them in the future.
