---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Using the points layer

In this document, you will learn about the `napari` `Points` layer, including
displaying spots over an image that have been found in an automated fashion, or
manually annotating an image with points. You will also understand how to add a
points layer and edit it from the GUI and from the console.

## When to use the points layer

The points layer allows you to display an `NxD` array of `N` points in `D`
coordinates. You can adjust the size, face color, and edge color of all the
points independently. You can also adjust the opactiy, edge width, and symbol
representing all the points simultaneously.

Each data point can have annotations associated with it using the
`Points.properties` dictionary. These properties can be used to set the face and
edge colors of the points. For example, when displaying points of different
classes/types, one could automatically set color the individual points by their
respective class/type. For more details on point properties, see the "setting
point edge and face color with properties" below or the [point annotation
tutorial](../../tutorials/annotation/annotate_points).

## A simple example

You can create a new viewer and add a set of points in one go using the
`napari.view_points` method, or if you already have an existing viewer, you can
add points to it using `viewer.add_points`. The api of both methods is the same.
In these examples we'll mainly use `add_points` to overlay points onto on an
existing image.

In this example, we will overlay some points on the image of an astronaut:

```{code-cell} python
import napari
import numpy as np
from skimage import data

viewer = napari.view_image(data.astronaut(), rgb=True)
points = np.array([[100, 100], [200, 200], [300, 100]])

points_layer = viewer.add_points(points, size=30)
```

```{code-cell} python
:tags: [hide-input]

from napari.utils import nbscreenshot

nbscreenshot(viewer, alt_text="3 points overlaid on an astronaut image")
```

```{code-cell} python
:tags: [remove-cell]

viewer.close()
```

## Arguments of `view_points` and `add_points`

{meth}`~napari.view_layers.view_points` and {meth}`~napari.Viewer.add_points`
accept the same layer-creation parameters.

```{code-cell} python
:tags: [hide-cell]

help(napari.view_points)
```

## Points data

The input data to the points layer must be an `NxD` numpy array containing the
coordinates of `N` points in `D` dimensions. The ordering of these dimensions is
the same as the ordering of the dimensions for image layers. This array is
always accessible through the `layer.data` property and will grow or shrink as
new points are either added or deleted.

## Using the points properties dictionary

The `Points` layer can contain properties that annotate each point.
`Points.properties` stores the properties in a dictionary where each key is the
name of the property and the values are numpy arrays with a value for each point
(i.e., length N for N points in `Points.data`). As we will see below, we can use
the values in a property to set the display properties of the points (e.g., face
color or edge color). To see the points properties in action, please see the
[point annotation tutorial](../../tutorials/annotation/annotate_points).

## Creating a new points layer

As you can add new points to a points layer using the add points tool, it is
possible to create a brand new empty points layers by clicking the new points
layer button above the layers list. The shape of the points layer is defined by
the points inside it, and so as you add new points the shape will adjust as
needed. The dimension of the new points layer will default to the largest
dimension of any layer currently in the viewer, or to 2 if no other layers are
present in the viewer.

## Non-editable mode

If you want to disable editing of the points layer you can set the `editable`
property of the layer to `False`.

As note in the section on 3D rendering, when using 3D rendering the points layer
is not editable.

## 3D rendering of points

All our layers can be rendered in both 2D and 3D mode, and one of our viewer
buttons can toggle between each mode. The number of dimensions sliders will be 2
or 3 less than the total number of dimensions of the layer. See for example
these points overlaid on an image in both 2D and 3D:

![image: smFISH with points overlaid](../../images/smFISH.webm)

Note though that when entering 3D rendering mode the point add, delete, and
select tools are all disabled. Those options are only supported when viewing a
layer using 2D rendering.

## Pan and zoom mode

The default mode of the points layer is to support panning and zooming, as in
the image layer. This mode is represented by the magnifying glass in the layers
control panel, and while it is selected editing the layer is not possible.
Continue reading to learn how to use some of the editing modes. You can always
return to pan and zoom mode by pressing the `Z` key when the points layer is
selected.

## Adding, deleting, and selecting points

New points can be added using the point adding tool. This tool can be selected
from layer controls panel. Points can then be added by clicking in the canvas.
If you have a multidimensional points layer then the coordinates of the new
point will keep track of the currently viewed slice that you added the point
too. You can quickly select the add points tool by pressing the `P` key when the
points layer is selected. The point adding tool also supports panning and
zooming.

You can select a point by selecting the select points tool and then clicking on
that point. You can select multiple points by continuing to shift click on
additional points, or by dragging a bounding box around the points you want to
select. You can quickly select the select points tool by pressing the `S` key
when the points layer is selected.

You can select all the points in the currently viewed slice by clicking the `A`
key if you are in select mode.

Once selected you can delete the selected points by clicking on the delete
button in the layer controls panel or pressing the delete key.

When using the point selection tool the pan and zoom functionality of the viewer
canvas is disabled and you are able to select points the layer. You can
temporarily re-enable pan and zoom by pressing and holding the spacebar. This
feature can be useful if you want to move around the points layer as you create
your selection.

## Changing points size

Each point can have a different size. You can pass a list or 1-dimensional array
of points through the size keyword argument to initialize the layer with points
of different sizes. These sizes are then accessible through the `size`
property. If you pass a single size then all points will get initialized with
that size. Points can be pseudo-visualized as n-dimensional if the
`out_of_slice_display` property is set to `True` or the `out of slice` checkbox is
checked. In this setting when viewing different slices of the layer points will
appear in the neighbouring slices to the ones in which they are located with a
size scaled by the distance from their center to that slice. This feature can be
especially useful when visualizing 2D slices of points that are located in a 3D
volume.

Points can also be resized within the GUI by first selecting them and then
adjusting the point size slider. If no points are selected, then adjusting the
slider value will only serve to initialize the size for new points that are
about to be added. The value of the size of the next point to be added can be
found in the `layer.current_size` property. Note this property is different from
`layer.size` which contains the current sizes of all the points.

## Changing points edge and face color

Individual points can each have different edge and face colors. You can
initially set these colors by providing a list of colors to the `edge_color` or
`face_color` keyword arguments respectively, or you can edit them from the GUI.
The colors of each of the points are available as lists under the
`layer.edge_color` and `layer.face_color` properties. Similar to the `size`
and `current_size` properties these properties are different from the
`layer.current_edge_color` and `layer.current_face_color` properties that will
determine the color of the next point to be added or any currently selected
points.

To change the point color properties from the GUI you must first select the
points whose properties you want to change, otherwise you will just be
initializing the property for the next point you add.

## Setting point edge and face color with properties

Point edge and face colors can be set as a function of a property in
`Points.properties`. There are two ways that the values in properties can be
mapped to colors: (1) color cycles and (2) colormaps.

Color cycles are sets of colors that are mapped to categorical properties. The
colors are repeated if the number of unique property values is greater than the
number of colors in the color cycle.

Colormaps are a continuum of colors that are mapped to a continuous property
value. The available colormaps are listed below (colormaps are from
[vispy](https://vispy.org/api/vispy.color.colormap.html#vispy.color.colormap.Colormap)).
For some guidance on choosing colormaps, see the [matplotlib colormap
docs](https://matplotlib.org/stable/tutorials/colors/colormaps.html).

```{code-cell} python
list(napari.utils.colormaps.AVAILABLE_COLORMAPS)
```

### Setting edge or face color with a color cycle

Here we will set the edge color of the markers with a color cycle on a property.
To do the same for a face color, substitute `face_color` for `edge_color` in the
example snippet below.

```{code-cell} python
viewer = napari.view_image(data.astronaut(), rgb=True)
points = np.array([[100, 100], [200, 200], [300, 100]])
point_properties = {
    'good_point': np.array([True, True, False]),
    'confidence': np.array([0.99, 0.8, 0.2]),
}

points_layer = viewer.add_points(
    points,
    properties=point_properties,
    edge_color='good_point',
    edge_color_cycle=['magenta', 'green'],
    edge_width=0.5,
)
```

```{code-cell} python
:tags: [hide-input]
nbscreenshot(viewer, alt_text="3 points overlaid on an astronaut image, where the edge color of the points has been changed to a color cycle")
```

```{code-cell} python
:tags: [remove-cell]

viewer.close()
```

In the example above, the properties (`point_properties`) were provided as a
dictionary with two properties: `good_point` and `confidence`. The values of
each property are stored in a numpy ndarray with length 3 since there were three
coordinates provided in `points`. We set the edge color as a function of the
`good_point` property by providing the keyword argument
`edge_color='good_point'` to the `viewer.add_points()` method. We set the color
cycle via the `edge_color_cycle` keyword argument (`edge_color_cycle=['magenta',
'green']`). The color cycle can be provided as a list of colors (a list of
strings or a (M x 4) array of M RGBA colors).

### Setting edge or face color with a colormap

Here we will set the face color of the markers with a colormap on a property.
To do the same for an edge color, substitute `face` for `edge` in the
example snippet below.

```{code-cell} python
viewer = napari.view_image(data.astronaut(), rgb=True)
points = np.array([[100, 100], [200, 200], [300, 100]])
point_properties = {
    'good_point': np.array([True, True, False]),
    'confidence': np.array([0.99, 0.8, 0.2]),
}

points_layer = viewer.add_points(
    points,
    properties=point_properties,
    face_color='confidence',
    face_colormap='viridis',
)
```

```{code-cell} python
:tags: [hide-input]
nbscreenshot(viewer, alt_text="3 points overlaid on an astronaut image, where the face color of the points has been changed to a colormap")
```

```{code-cell} python
:tags: [remove-cell]

viewer.close()
```

In the example above, the properties (`point_properties`) were provided as a
dictionary with two properties: `good_point` and `confidence`. The values of
each property are stored in a numpy ndarray with length 3 since there were three
coordinates provided in `points`. We set the face color as a function of the
`confidence` property by providing the keyword argument
`face_color='confidence'` to the `viewer.add_points()` method. We set the
colormap to viridis using the `face_colormap` keyword argument
(`face_colormap='viridis'`).

## Changing the points symbol

The symbol for the points layer is a global property for the layer. All points
must have the same symbol. You can set the symbol on the loading of the layer
using the `symbol` keyword argument, or you can change it from the the GUI using
the symbol dropdown menu. Since the symbol property applies to all the points
you don't need to have any points selected for it to have an effect.

## Copying and pasting points

It is possible to copy and paste any selected points using the `ctrl-C` and
`ctrl-V` keybindings, respectively. If you have a multidimensional `Points`
layer you can copy points from one slice to another by pasting them into the new
slice. The coordinates of the points in the visible dimensions will be in the
same place on the new slice as in the old slice, but the rest of the coordinates
will be updated with the new slice values.

## Putting it all together

Here you can see an example of adding, selecting, deleting points and change
their properties:

![image: editing points](../../images/editing_points.webm)
