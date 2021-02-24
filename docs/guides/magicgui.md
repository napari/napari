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

For more information on the features and usage of `magicgui`, see the [magicgui
documentation](https://napari.org/magicgui).  `magicgui` does not require
napari, but napari *does* provide support for using magicgui within napari. The
purpose of this page is to document some of the conveniences provided by napari
when using `magicgui` with napari-specific type annotations.

## magicgui and type annotations

`magicgui` uses [type hints](https://www.python.org/dev/peps/pep-0484/) to infer
the appropriate widget type to display a given function parameter (or, in the
absense of a type hint, the type of the default value may be used).  Third party
packages (like `napari` in this case) may provide support for their types using
[`magicgui.register_type`](https://napari.org/magicgui/usage/types_widgets.html#register-type).
This is how using the type annotations described below lead to "actions" in napari.

```{important}
All of the type annotations described below *require* that the resulting widget
be added to a napari viewer (using, e.g., `viewer.window.add_dock_widget`, or
providing a magicgui-based widget via the {func}`~napari.plugins.hook_specifications.napari_experimental_provide_dock_widget` plugin hookspec)
```

## Getting information from napari into your magicgui function

The following napari types may be used as *parameter* type annotations in magicgui
functions. The consequence of each is described below:

- any napari {class}`~napari.layers.Layer` type, such as
  {class}`~napari.layers.Image` or {class}`~napari.layers.Points`
- any of the `<Layer>Data` types from {mod}`napari.types`, such as
  {attr}`napari.types.ImageData` or  {attr}`napari.types.LabelsData`
- {class}`napari.Viewer`

### Annotating parameters as a `Layer` subclasses

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
nbscreenshot(viewer)
```

### Annotating parameters as `Layer`

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

### Annotating parameters as a `napari.types.<LayerType>Data`

In the previous example, the object passed to your function will be the actual
{class}`~napari.layers.Layer` instance, meaning you will need to access any
attributes (like `layer.data`) on your own.  If your function is designed to
accept a numpy array, you can use the any of the special `<Layer>Data` types
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

### Annotating parameters as a `napari.Viewer`

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
described [below](#adding-layers-to-napari-from-your-magicgui-function).
If you find that you require the viewer instance because of functionality that
is otherwise missing here, please consider opening an issue in the
[napari issue tracker](https://github.com/napari/napari/issues/new/choose),
describing your use case.
```

## Adding layers to napari from your magicgui function

The following napari types may be used as *return* type annotations in magicgui
functions. The consequence of each is described below:

- any napari {class}`~napari.layers.Layer` type, such as
  {class}`~napari.layers.Image` or {class}`~napari.layers.Points`
- any of the `<Layer>Data` types from {mod}`napari.types`, such as
  {attr}`napari.types.ImageData` or  {attr}`napari.types.LabelsData`
- {attr}`napari.types.LayerDataTuple`

### Return annotation of `Layer` subclass

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
nbscreenshot(viewer)
```

```{note}
With this method, a new layer will be added to the layer list each time the
function is called.  To update an existing layer, you must use the
`LayerDataTuple` approach described below
```

### Return annotation of `napari.types.<LayerType>Data`

In the previous example, the object returned by the function had to be an actual
{class}`~napari.layers.Layer` instance (in keeping with the return type
annotation).  In many cases, you may only be interested in receiving and
returning the layer {attr}`~napari.layers.Layer.data`  itself.  (There are
*many* functions already written that accept and return a `numpy.ndarray`, for
example). In this case, you may use a return type annotation of one the special
`<Layer>Data` types from {mod}`napari.types` to indicate that you want data
returned by your function to be turned into the corresponding
{class}`~napari.layers.Layer` type, and added to the viewer.

For example, in combination with the {attr}`~napari.types.ImageData`` paramater
annotation [described
above](#annotating-parameters-as-a-naparitypeslayertypedata):

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
nbscreenshot(viewer)
```

### Return annotation of `napari.types.<LayerType>Data`

The most flexible return type annotation is {attr}`napari.types.LayerDataTuple`:
it gives you full control over the layer that will be created and added to the
viewer.  (It also lets you update an existing layer with a matching name).

A {attr}`~napari.types.LayerDataTuple` is a {class}`tuple` in one of the
following three forms:

1. data only: `(layer_data,)`
2. data and metadata {class}`dict`: `(layer_data, {})`
   - the metadata `dict` is a key-value mapping, where the keys must be valid keyword
     arguments to the corresponding {class}`napari.layers.Layer` constructor.
3. data, metadata, and layer type string: `(layer_data, {}, 'layer_type')`.
   - `layer_type` should be a lowercase string form of one of the layer types
     (like `'points'`, `'shapes'`, etc...).  If omitted, the layer type is assumed
     to be `'image'`.
