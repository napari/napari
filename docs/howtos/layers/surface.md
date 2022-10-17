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

# Using the surface layer

In this document, you will learn about the `napari` `Surface` layer, including
how to display surface data and edit the properties of surfaces like the
contrast, opacity, colormaps and blending mode. You will also understand how to
add and manipulate surfaces both from the GUI and from the console.

## When to use the surface layer

The surface layer allows you to display a precomputed surface mesh that is
defined by an `NxD` array of `N` vertices in `D` coordinates, an `Mx3` integer
array of the indices of the triangles making up the faces of the surface, and a
length `N` list of values to associate with each vertex to use alongside a
colormap.

## A simple example

You can create a new viewer and add a surface in one go using the
`napari.view_surface` method, or if you already have an existing viewer, you can
add an image to it using `viewer.add_surface`. The api of both methods is the
same. In these examples we'll mainly use `view_surface`.

A simple example of viewing a surface is as follows:

```{code-cell} python
import napari
import numpy as np

vertices = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
faces = np.array([[0, 1, 2], [1, 2, 3]])
values = np.linspace(0, 1, len(vertices))
surface = (vertices, faces, values)

viewer = napari.view_surface(surface)  # add the surface
```

```{code-cell} python
:tags: [hide-input]

from napari.utils import nbscreenshot

nbscreenshot(viewer, alt_text="A viewer with a surface")
```

```{code-cell} python
:tags: [remove-cell]

viewer.close()
```

## Arguments of `view_surface` and `add_surface`

{meth}`~napari.view_layers.view_surface` and {meth}`~napari.Viewer.add_surface`
accept the same layer-creation parameters.

```{code-cell} python
:tags: [hide-cell]

help(napari.view_surface)
```

## Surface data

The data for a surface layer is defined by a 3-tuple of its vertices, faces, and
vertex values. The vertices are an `NxD` array of `N` vertices in `D`
coordinates. The faces are an `Mx3` integer array of the indices of the
triangles making up the faces of the surface. The vertex values are a length `N`
list of values to associate with each vertex to use alongside a colormap. This
3-tuple is accessible through the `layer.data` property.

## 3D rendering of images

All our layers can be rendered in both 2D and 3D mode, and one of our viewer
buttons can toggle between each mode. The number of dimensions sliders will be 2
or 3 less than the total number of dimensions of the layer. See for example
these brain surfaces rendered in 3D:

![image: brain surface](../../images/brain_surface.webm)

## Working with colormaps

The same colormaps available for the `Image` layer are also available for the
`Surface` layer. napari supports any colormap that is created with
`vispy.color.Colormap`. We provide access to some standard colormaps that you
can set using a string of their name.

```{code-cell} python
list(napari.utils.colormaps.AVAILABLE_COLORMAPS)
```

Passing any of these as follows as keyword arguments will set the colormap of
that surface. You can also access the current colormap through the
`layer.colormap` property which returns a tuple of the colormap name followed by
the vispy colormap object. You can list all the available colormaps using
`layer.colormaps`.

It is also possible to create your own colormaps using vispy's
`vispy.color.Colormap` object, see it's full
[documentation here](https://vispy.org/api/vispy.color.colormap.html#vispy.color.colormap.Colormap).
For more detail see the [image layer guide](./image).

## Adjusting contrast limits

The vertex values of the surface layer get mapped through its colormap according
to values called contrast limits. These are a 2-tuple of values defining how
what values get applied the minimum and maximum of the colormap and follow the
same principles as the `contrast_limits` described in the [image layer
guide](./image). They are also accessible through the same keyword arguments,
properties, and range slider as in the image layer.
