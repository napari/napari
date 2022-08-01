---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
(magicgui)=

# Using `magicgui` in napari

## magicgui

[magicgui](https://github.com/napari/magicgui) is a python package that assists
in building small, composable graphical user interfaces (widgets). It is a general
abstraction layer on GUI toolkit backends (like Qt), with an emphasis on mapping
python types to widgets.  In particular, it makes building widgets to represent
function inputs easy:

```{code-cell} python
from magicgui import magicgui
import datetime
import pathlib

@magicgui(
    call_button="Calculate",
    slider_float={"widget_type": "FloatSlider", 'max': 10},
    dropdown={"choices": ['first', 'second', 'third']},
)
def widget_demo(
    maybe: bool,
    some_int: int,
    spin_float=3.14159,
    slider_float=4.5,
    string="Text goes here",
    dropdown='first',
    date=datetime.datetime.now(),
    filename=pathlib.Path('/some/path.ext')
):
    ...

widget_demo.show()
```
*Add caption here*

For more information on the features and usage of `magicgui`, see the [magicgui
documentation](https://napari.org/magicgui).  `magicgui` does not require
napari, but napari *does* provide support for using magicgui within napari. The
purpose of this page is to document some of the conveniences provided by napari
when using `magicgui` with napari-specific type annotations.

## magicgui and type annotations

`magicgui` uses [type hints](https://peps.python.org/pep-0484/) to infer
the appropriate widget type for a given function parameter, and to indicate a
context-dependent action for the object returned from the function (in the
absence of a type hint, the type of the default value will be used).  Third
party packages (like `napari` in this case) may provide support for their types
using
[`magicgui.register_type`](https://napari.org/magicgui/usage/types_widgets.html#register-type).
This is how using the type annotations described below leads to widgets and/or
"actions" in napari.

```{important}
All of the type annotations described below *require* that the resulting widget
be added to a napari viewer (either via `viewer.window.add_dock_widget`, or
by providing a magicgui-based widget via the {func}`~napari.plugins.hook_specifications.napari_experimental_provide_dock_widget` plugin hook specification).
```

## Parameter annotations

The following napari types may be used as *parameter* type annotations in
magicgui functions to get information from the napari viewer into your
magicgui function. The consequence of each type annotation is described below:

- any napari {class}`~napari.layers.Layer` type, such as
  {class}`~napari.layers.Image` or {class}`~napari.layers.Points`
- any of the `<LayerType>Data` types from {mod}`napari.types`, such as
  {attr}`napari.types.ImageData` or  {attr}`napari.types.LabelsData`
- {class}`napari.Viewer`

### Annotating as a `Layer` subclass

If you annotate one of your function parameters as a
{class}`~napari.layers.Layer` subclass (such as {class}`~napari.layers.Image` or
{class}`~napari.layers.Points`), it will be rendered as a
{class}`~magicgui.widgets.ComboBox` widget (i.e. "dropdown menu"), where the
options in the dropdown box are the layers of the corresponding type currently
in the viewer.

```python
from napari.layers import Image

@magicgui
def my_widget(image: Image):
    # do something with whatever image layer the user has selected
    # note: it *may* be None! so your function should handle the null case
    ...
```

Here's a complete example:

```{code-cell} python
:tags: [remove-output]
import napari
import numpy as np
from napari.layers import Image

@magicgui(image={'label': 'Pick an Image'})
def my_widget(image: Image):
    ...

viewer = napari.view_image(np.random.rand(64, 64), name="My Image")
viewer.window.add_dock_widget(my_widget)
```
*Note the widget at the bottom with "My Image" as the currently selected option*

```{code-cell} python
:tags: [remove-input]
from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
nbscreenshot(viewer, alt_text="A magicgui widget using an image layer parameter annotation")
```

### Annotating as `Layer`

In the previous example, the dropdown menu will *only* show
{class}`~napari.layers.Image` layers, because the parameter was annotated as an
{class}`~napari.layers.Image`.  If you'd like a dropdown menu that allows the
user to pick from *all* layers in the layer list, annotate your parameter as
{class}`~napari.layers.Layer`

```python
from napari.layers import Layer

@magicgui
def my_widget(layer: Layer):
    # do something with whatever layer the user has selected
    # note: it *may* be None! so your function should handle the null case
    ...
```

(annotating-as-napari-types-data)=
### Annotating as `napari.types.*Data`

In the previous example, the object passed to your function will be the actual
{class}`~napari.layers.Layer` instance, meaning you will need to access any
attributes (like `layer.data`) on your own.  If your function is designed to
accept a numpy array, you can use any of the special `<LayerType>Data` types
from {mod}`napari.types` to indicate that you only want the data attribute from
the layer (where `<LayerType>` is one of the available layer types).  Here's an
example using {attr}`napari.types.ImageData`

```python
from napari.types import ImageData
import numpy as np

@magicgui
def my_widget(array: ImageData):
    # note: it *may* be None! so your function should handle the null case
    if array is not None:
      assert isinstance(array, np.ndarray)  # it will be!
```

### Annotating as `napari.Viewer`

Lastly, if you need to access the actual {class}`~napari.viewer.Viewer` instance
in which the widget is docked, you can annotate one of your parameters as a
{class}`napari.Viewer`.

```python
from napari import Viewer

@magicgui
def my_widget(viewer: Viewer):
  ...
```

```{caution}
Please use this sparingly, as a last resort. If you need to *add* layers
to the viewer from your function, prefer one of the return-annotation methods
described [below](#return-annotations).
If you find that you require the viewer instance because of functionality that
is otherwise missing here, please consider opening an issue in the
[napari issue tracker](https://github.com/napari/napari/issues/new/choose),
describing your use case.
```

## Return annotations

The following napari types may be used as *return* type annotations in `magicgui`
functions to add layers to napari from your `magicgui` function. The consequence of
each type is described below:

- any napari {class}`~napari.layers.Layer` type, such as
  {class}`~napari.layers.Image` or {class}`~napari.layers.Points`
- any of the `<LayerType>Data` types from {mod}`napari.types`, such as
  {attr}`napari.types.ImageData` or  {attr}`napari.types.LabelsData`
- {attr}`napari.types.LayerDataTuple`
- `List`s of {class}`napari.layers.Layer` or {attr}`napari.types.LayerDataTuple`

### Returning a `Layer` subclass

If you use a {class}`~napari.layers.Layer` subclass as a *return* annotation on a
`magicgui` function, `napari` will interpet it to mean that the layer returned
from the function should be added to the viewer.  The object returned from the
function must be an actual {class}`~napari.layers.Layer` instance.

```python
from napari.layers import Image
import numpy as np

@magicgui
def my_widget(ny: int=64, nx: int=64) -> Image:
  return Image(np.random.rand(ny, nx), name='my Image')
```

Here's a complete example

```{code-cell} python
:tags: [remove-output]
@magicgui(call_button='Add Image')
def my_widget(ny: int=64, nx: int=64) -> Image:
  return Image(np.random.rand(ny, nx), name='My Image')

viewer = napari.Viewer()
viewer.window.add_dock_widget(my_widget, area='right')
my_widget()  # "call the widget" to call the function.
             # Normally this would be caused by some user UI interaction
```

*Note the new "My Image" layer in the viewer as a result of having called the widget function.*

```{code-cell} python
:tags: [remove-input]
from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
nbscreenshot(viewer, alt_text="A magicgui widget using an image layer return annotation")
```

```{note}
With this method, a new layer will be added to the layer list each time the
function is called.  To update an existing layer, you must use the
`LayerDataTuple` approach described below
```

### Returning `List[napari.layers.Layer]`

You can create multiple layers by returning a list of
{class}`~napari.layers.Layer`.

```python
from typing import List

@magicgui
def make_points(...) -> List[napari.layers.Layer]:
  ...
```

```{note}
Note: the `List[]` syntax here is optional from the perspective of `napari`.  You
can return either a single Layer or a list of Layers and they will all be added
to the viewer as long as you use either `List[napari.layers.Layer]` or 
`napari.layers.Layer`.  If you want your code to be properly typed, however,
your return type must match your return annotation.
```

(returning-napari-types-data)=
### Returning `napari.types.*Data`

In the previous example, the object returned by the function had to be an actual
{class}`~napari.layers.Layer` instance (in keeping with the return type
annotation).  In many cases, you may only be interested in receiving and
returning the layer {attr}`~napari.layers.Layer.data`  itself.  (There are
*many* functions already written that accept and return a `numpy.ndarray`, for
example). In this case, you may use a return type annotation of one the special
`<LayerType>Data` types from {mod}`napari.types` to indicate that you want data
returned by your function to be turned into the corresponding
{class}`~napari.layers.Layer` type, and added to the viewer.

For example, in combination with the {attr}`~napari.types.ImageData` paramater
annotation [described above](annotating-as-napari-types-data):

```{code-cell} python
:tags: [remove-output]
from napari.types import LabelsData, ImageData

@magicgui(call_button='Run Threshold')
def threshold(image: ImageData, threshold: int = 75) -> LabelsData:
    """Threshold an image and return a mask."""
    return (image > threshold).astype(int)

viewer = napari.view_image(np.random.randint(0, 100, (64, 64)))
viewer.window.add_dock_widget(threshold)
threshold()  # "call the widget" to call the function.
             # Normally this would be caused by some user UI interaction
```

```{code-cell} python
:tags: [remove-input]
from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
nbscreenshot(viewer, alt_text="A magicgui widget returning a layer attribute")
```

### Returning `napari.types.LayerDataTuple`

The most flexible return type annotation is {attr}`napari.types.LayerDataTuple`:
it gives you full control over the layer that will be created and added to the
viewer.  (It also lets you update an existing layer with a matching name).

A {attr}`~napari.types.LayerDataTuple` is a {class}`tuple` in one of the
following three forms:

1. `(layer_data,)`
   - a single item tuple containing only layer data (will be interpreted as an image).
2. `(layer_data, {})`
   - a 2-tuple of `layer_data` and a metadata {class}`dict`. the keys in the
     metadata `dict` must be valid keyword arguments to the corresponding
     {class}`napari.layers.Layer` constructor.
3. `(layer_data, {}, 'layer_type')`
   - a 3-tuple of data, metadata, and layer type string.`layer_type` should be a
     lowercase string form of one of the layer types (like `'points'`,
     `'shapes'`, etc...).  If omitted, the layer type is assumed to be
     `'image'`.

The following are all valid {attr}`napari.types.LayerDataTuple` examples:

```python
# an image array
(np.random.rand(64, 64),) 

# an image with name and custom blending mode
(np.random.rand(64, 64), {'name': 'My Image', 'blending': 'additive'})

# an empty points layer
(None, {}, 'points')

# points with properties
(np.random.rand(20, 2), {'properties': {'values': np.random.rand(20)}}, 'points')
```

An example of using a {attr}`~napari.types.LayerDataTuple` return annotation in
a `magicgui` function:

```{code-cell} python
:tags: [remove-output]
import napari.types

@magicgui(call_button='Make Points')
def make_points(n_points=40) -> napari.types.LayerDataTuple:
  data = 500 * np.random.rand(n_points, 2)
  props = {'values': np.random.rand(n_points)}
  return (data, {'properties': props}, 'points')

viewer = napari.Viewer()
viewer.window.add_dock_widget(make_points)
make_points()  # "call the widget" to call the function.
               # Normally this would be caused by some user UI interaction
```

```{code-cell} python
:tags: [remove-input]
from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
nbscreenshot(viewer, alt_text="A magicgui widget returning a LayerDataTuple")
```

### Returning `List[napari.types.LayerDataTuple]`

You can also create multiple layers by returning a list of
{attr}`~napari.types.LayerDataTuple`.

```python
from typing import List

@magicgui
def make_points(...) -> List[napari.types.LayerDataTuple]:
  ...
```

```{note}
Note: the `List[]` syntax here is optional from the perspective of `napari`.  You
can return either a single tuple or a list of tuples and they will all be added
to the viewer as long as you use either `List[napari.types.LayerDataTuple]` or 
`napari.types.LayerDataTuple`.  If you want your code to be properly typed, however,
your return type must match your return annotation.
```

### Updating an existing Layer

The default behavior is to add a new layer to the viewer for each
`LayerDataTuple` returned by a magicgui function. By providing a unique
`name` key in your {attr}`~napari.types.LayerDataTuple` metadata dict, you can
update an existing layer, rather than creating a new layer each time the
function is called:

```{code-cell} python
:tags: [remove-output]

@magicgui(call_button='Make Points', n_points={'maximum': 200})
def make_points(n_points=40) -> napari.types.LayerDataTuple:
  data = 500 * np.random.rand(n_points, 2)
  return (data, {'name': 'My Points'}, 'points')

viewer = napari.Viewer()
viewer.window.add_dock_widget(make_points)
# calling this multiple times will just update 'My Points'
make_points()
make_points.n_points.value = 80
make_points()
make_points.n_points.value = 120
make_points()
```

```{code-cell} python
:tags: [remove-input]
from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
nbscreenshot(viewer, alt_text="A magicgui widget updating an existing layer")
```

## Avoid imports with forward references

Sometimes, it is undesirable to import and/or depend on `napari` directly just
to provide type annotations.  It is possible to avoid importing `napari`
entirely by annotating with the string form of the napari type.  This is called
a [Forward
reference](https://peps.python.org/pep-0484/#forward-references):

```python
@magicgui
def my_func(data: 'napari.types.ImageData') -> 'napari.types.ImageData':
    ...
```

:::{tip}

If you'd like to maintain IDE type support and autocompletion, you can
do so by hiding the napari imports inside of a {attr}`typing.TYPE_CHECKING`
clause:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import napari

@magicgui
def my_func(data: 'napari.types.ImageData') -> 'napari.types.ImageData':
    ...
```

This will not require `napari` at runtime, but if it is installed in your
development environment, you will still get all the type inference.

:::

## Using `magicgui` in napari plugin widgets

Using `magicgui` can be an effective way to generate widgets for use in napari
{ref}`plugins-index`, in particular the
{func}`~napari.plugins.hook_specifications.napari_experimental_provide_dock_widget`
plugin hook specification.  There is an important distinction to be made,
however, between using `magicgui` with `viewer.window.add_dock_widget`, and
using it with
{func}`~napari.plugins.hook_specifications.napari_experimental_provide_dock_widget`.

`viewer.window.add_dock_widget` expects an *instance* of a widget, like a
{class}`magicgui.widgets.Widget` or a {class}`qtpy.QtWidgets.QWidget`.
{func}`~napari.plugins.hook_specifications.napari_experimental_provide_dock_widget`,
on the other hand, expects a widget *class* (or, more broadly, a `callable` that
returns a widget instance).  There are two ways to acheive this with `magicgui`.

### `@magic_factory`

In most cases, the {func}`@magicgui <magicgui.magicgui>` decorator used in the
preceding examples can simply be replaced with the {func}`@magic_factory <magicgui.magic_factory>`
decorator, to use it as a plugin dock widget.

For example, the threshold widget [shown above](returning-napari-types-data)
could be provided as a napari plugin as follows:

```python
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation

@magic_factory(auto_call=True, threshold={'max': 2 ** 16})
def threshold(
    data: 'napari.types.ImageData', threshold: int
) -> 'napari.types.LabelsData':
    return (data > threshold).astype(int)

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return threshold
```

:::{note}
{func}`@magic_factory <magicgui.magic_factory>` behaves very much like
{func}`functools.partial`: it returns a callable that "remembers" some or
all of the parameters required for a "future" call to {func}`magicgui.magicgui`.
The parameters provided to {func}`@magic_factory <magicgui.magic_factory>` can
also be overridden when creating a widget from a factory:

```python
@magic_factory(call_button=True)
def my_factory(x: int):
    ...

widget1 = my_factory()
widget2 = my_factory(call_button=False, x={'widget_type': 'Slider'})
```

:::

### `magicgui.widgets.FunctionGui`

The other option for using `magicgui` in plugins is to directly subclass
{class}`magicgui.widgets.FunctionGui` (which is the type that is returned
by the {func}`@magicgui <magicgui.magicgui>` decorator).

```python
from magicgui.widgets import FunctionGui

def my_function(...):
    ...

class MyGui(FunctionGui):
    def __init__(self):
        super().__init__(
          my_function,
          call_button=True,
          layout='vertical',
          param_options={...}
        )
        # do whatever other initialization you want here

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MyGui
```
