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

# Using the vectors layer

In this document, you will learn about the `napari` `Vectors` layer, including
how to display many vectors simultaneously and adjust their properties. You will
also understand how to add a vectors layer and edit it from the GUI and from the
console.

## When to use the vectors layer

The vectors layer allows you to display many vectors with defined starting
points and directions. It is particularly useful for people who want to
visualize large vector fields, for example if you are doing polarization
microscopy. You can adjust the color, width, and length of all the vectors both
programmatically and from the GUI.

## A simple example

You can create a new viewer and add vectors in one go using the
`napari.view_vectors` method, or if you already have an existing viewer, you can
add shapes to it using `viewer.add_vectors`. The api of both methods is the
same. In these examples we'll mainly use `add_vectors` to overlay shapes onto on
an existing image.

In this example, we will overlay some shapes on the image of a photographer:

```{code-cell} python
import napari
import numpy as np
from skimage import data

# create vector data
n = 250
vectors = np.zeros((n, 2, 2), dtype=np.float32)
phi_space = np.linspace(0, 4 * np.pi, n)
radius_space = np.linspace(0, 100, n)
# assign x-y projection
vectors[:, 1, 0] = radius_space * np.cos(phi_space)
vectors[:, 1, 1] = radius_space * np.sin(phi_space)
# assign x-y position
vectors[:, 0] = vectors[:, 1] + 256

# add the image
viewer = napari.view_image(data.camera(), name='photographer')
# add the vectors
vectors_layer = viewer.add_vectors(vectors, edge_width=3)
```

```{code-cell} python
:tags: [hide-input]

from napari.utils import nbscreenshot

nbscreenshot(viewer, alt_text="Vectors overlaid on an image")
```

```{code-cell} python
:tags: [remove-cell]

viewer.close()
```

## Arguments of `view_vectors` and `add_vectors`

{meth}`~napari.view_layers.view_vectors` and {meth}`~napari.Viewer.add_vectors`
accept the same layer-creation parameters.

```{code-cell} python
:tags: [hide-cell]

help(napari.view_vectors)
```

## Vectors data

The input data to the vectors layer must either be a `Nx2xD` numpy array
representing `N` vectors with start position and projection values in `D`
dimensions, or it must be an `N1xN2 ... xNDxD`, array where each of the first
`D` dimensions corresponds to the voxel of the location of the vector, and the
last dimension contains the `D` values of the projection of that vector. The
former representation is useful when you have vectors that can start in
arbitrary positions in the canvas. The latter representation is useful when your
vectors are defined on a grid, say corresponding to the voxels of an image, and
you have one vector per grid.

See here for the example from
[`examples/add_vectors_image.py`](https://github.com/napari/napari/blob/main/examples/add_vectors_image.py)
of a grid of vectors defined over a random image:

![image: add vectors overlaid on an image ](../../images/add_vectors_image.png)

Regardless of how the data is passed, we convert it to the `Nx2xD`
representation internally. This representation is  accessible through the
`layer.data` property.

Editing the start position of the vectors from the GUI is not possible. Nor is
it possible to draw vectors from the GUI. If you want to draw lines from the GUI
you should use the `Lines` shape inside a `Shapes` layer.

## 3D rendering of vectors

All our layers can be rendered in both 2D and 3D mode, and one of our viewer
buttons can toggle between each mode. The number of dimensions sliders will be 2
or 3 less than the total number of dimensions of the layer. See for example the
[`examples/nD_vectors.py`](https://github.com/napari/napari/blob/main/examples/nD_vectors.py)
to see shapes in both 2D and 3D:

![image: nD vectors ](../../images/nD_vectors.webm)

## Changing vector length, width, and color

You can multiplicatively scale the length of all the vectors projections using
the `layer.length` property or combobox inside the layer controls panel.

You can also set the width of all the vectors in a layer using the `layer.width`
property or combobox inside the layer controls panel.

You can also set the color of all the vectors in a layer using the
`layer.edge_color` property or dropdown menu inside the layer controls panel.
