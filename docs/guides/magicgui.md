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

### Annotating parameters with `Layer` subclasses

If you annotated one of your function parameters with a
{class}`~napari.layers.Layer` subclass, it will be rendered as a
{class}`~magicgui.widgets.ComboBox` (i.e. "dropdown menu"), where the options in
the dropdown box are the layers of the corresponding type currently in the
viewer.

```python
from napari.layers import Image

@magicgui
def my_widget(layer: Image):
    ...
```

```{code-cell} python
import napari
from napari.layers import Points

@magicgui
def my_widget(layer: Points):
    ...

viewer = napari.Viewer()
viewer.window.add_dock_widget(my_widget)

```

## Adding layers to napari from your magicgui function

The following napari types may be used as *return* type annotations in magicgui
functions. The consequence of each is described below:

- any napari {class}`~napari.layers.Layer` type, such as
  {class}`~napari.layers.Image` or {class}`~napari.layers.Points`
- any of the `<Layer>Data` types from {mod}`napari.types`, such as
  {attr}`napari.types.ImageData` or  {attr}`napari.types.LabelsData`
- {attr}`napari.types.LayerDataTuple`