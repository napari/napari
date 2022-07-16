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

# Using the labels layer

In this document, you will learn about the `napari` `Labels` layer, including
using the layer to display the results of image segmentation analyses, and how
to manually segment images using the paintbrush and fill buckets. You will also
understand how to add a labels image and edit it from the GUI and from the
console.

## When to use the labels layer

The labels layer allows you to take an array of integers and display each
integer as a different random color, with the background color 0 rendered as
transparent.

The `Labels` layer is therefore especially useful for segmentation tasks where
each pixel is assigned to a different class, as occurs in semantic segmentation,
or where pixels corresponding to different objects all get assigned the same
label, as occurs in instance segmentation.

## A simple example

You can create a new viewer and add an labels image in one go using the
`napari.view_labels` method, or if you already have an existing viewer, you can
add a `Labels` image to it using `viewer.add_labels`. The api of both methods is
the same. In these examples we'll mainly use `add_labels` to overlay a `Labels`
image onto on image.

In this example of instance segmentation, we will find and segment each of the
coins in an image, assigning each one an integer label, and then overlay the
results on the original image as follows:

```{code-cell} python
import napari
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects

coins = data.coins()[50:-50, 50:-50]
# apply threshold
thresh = threshold_otsu(coins)
bw = closing(coins > thresh, square(4))
# remove artifacts connected to image border
cleared = remove_small_objects(clear_border(bw), 20)
# label image regions
label_image = label(cleared)

# create the viewer and add the coins image
viewer = napari.view_image(coins, name='coins')
# add the labels
labels_layer = viewer.add_labels(label_image, name='segmentation')
```

```{code-cell} python
:tags: [hide-input]

from napari.utils import nbscreenshot

nbscreenshot(viewer, alt_text="Segmentation of coins in an image, displayed using a labels layer")
```

```{code-cell} python
:tags: [remove-cell]

viewer.close()
```

## Arguments of `view_labels` and `add_labels`

{meth}`~napari.view_layers.view_labels` and {meth}`~napari.Viewer.add_labels`
accept the same layer-creation parameters.

```{code-cell} python
:tags: [hide-cell]

help(napari.view_labels)
```

## Labels data

The labels layer is a subclass of the `Image` layer and as such can support the
same numpy-like arrays, including
[dask arrays](https://docs.dask.org/en/stable/array.html),
[xarrays](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html),
and [zarr arrays](https://zarr.readthedocs.io/en/stable/api/core.html). A
`Labels` layer though must be integer valued, and the background label must be
0.

Because the labels layer subclasses the image layer it inherits the great
properties of the image layer, like supporting lazy loading and multiscale
images for big data layers. For more information about both these concepts see
the details in the [image layer guide](./image).

## Creating a new labels layer

As you can edit a `Labels` layer using the paintbrush and fill bucket, it is
possible to create a brand-new empty labels layers by clicking the new labels
layer button above the layers list. The shape of the new labels layer will match
the size of any currently existing image layers, allowing you to paint on top of
them.

## Saving labels layers

When saving a labels layer, file size can be unnecessarily large when compression
is not used.  By default, when saving a layer through the GUI no file 
compression is applied.  The following commands will save a labels layer using 
compression:

```{code-cell} 
from skimage.io import imsave
data = viewer.layers['name_of_your_layer'].data
imsave('path.tif', data, compress=1)
```

where `name_of_your_layer` represents the name of the layer being saved.  These commands
can be implemented in a script or entered at an ipython command prompt.

## Non-editable mode

If you want to disable editing of the labels layer you can set the `editable`
property of the layer to `False`.

As note in the section on 3D rendering, when using 3D rendering the labels layer
is not editable. Similarly, for now, a labels layer where the data is
represented as a multiscale image is not editable.

## 3D rendering of labels

All our layers can be rendered in both 2D and 3D mode, and one of our viewer
buttons can toggle between each mode. The number of dimensions sliders will be 2
or 3 less than the total number of dimensions of the layer, allowing you to
browse volumetric timeseries data and other high dimensional data.

```{code-cell} python
:tags: [remove-output]
from scipy import ndimage as ndi

blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
viewer = napari.view_image(blobs.astype(float), name='blobs')
labeled = ndi.label(blobs)[0]
labels_layer = viewer.add_labels(labeled, name='blob ID')
viewer.dims.ndisplay = 3
```

```{code-cell} python
:tags: [hide-input]

# programmatically adjust the camera angle
viewer.camera.zoom = 2
viewer.camera.angles = (3, 38, 53)
nbscreenshot(viewer, alt_text="A 3D view of a labels layer on top of 3D blobs")
```

Note though that when entering 3D rendering mode the colorpicker, paintbrush,
and fill bucket options which allow for layer editing are all disabled. Those
options are only supported when viewing a layer using 2D rendering.

## Pan and zoom mode

The default mode of the labels layer is to support panning and zooming, as in
the image layer. This mode is represented by the magnifying glass in the layers
control panel, and while it is selected editing the layer is not possible.
Continue reading to learn how to use some of the editing modes. You can always
return to pan and zoom mode by pressing the `Z` key when the labels layer is
selected.

## Shuffling label colors

The color that each integer gets assigned is random, aside from 0 which always
gets assigned to be transparent. The colormap we use is designed such that
nearby integers get assigned distinct colors. The exact colors that get assigned
as determined by a
[random seed](https://numpy.org/doc/stable/reference/generated/numpy.random.seed.html)
and changing that seed will shuffle the colors that each label gets assigned.
Changing the seed can be done by clicking on the `shuffle colors` button in the
layers control panel. Shuffling colors can be useful as some colors may be hard
to distinguish from the background or nearby objects.

## Selecting a label

A particular label can be selected either using the label combobox inside the
layers control panel and typing in the value of the desired label or using the
plus / minus buttons, or by selecting the color picker tool and then clicking on
a pixel with the desired label in the image. When a label is selected you will
see its integer inside the label combobox and the color or the label shown in
the thumbnail next to the label combobox. If the 0 label is selected, then a
checkerboard pattern is shown in the thumbnail to represent the transparent
color.

You can quickly select the color picker by pressing the `L` key when the labels
layer is selected.

You can set the selected label to be 0, the background label, by pressing `E`.

You can set the selected label to be one larger than the current largest label
by pressing `M`. This selection will guarantee that you are then using a label
that hasn't been used before.

You can also increment or decrement the currently selected label by pressing the
`I` or `D` key, respectively.

## Painting in the labels layer

One of the major use cases for the labels layer is to manually edit or create
image segmentations. One of the tools that can be used for manual editing is the
`paintbrush`, that can be made active from by clicking the paintbrush icon in
the layers control panel. Once the paintbrush is enabled, the pan and zoom
functionality of the  viewer canvas gets disabled, and you can paint onto the
canvas. You can temporarily re-enable pan and zoom by pressing and holding the
spacebar. This feature can be useful if you want to move around the labels layer
as you paint.

When you start painting the label that you are painting with, and the color that
you will see are determined by the selected label. Note there is no explicit
eraser tool, instead you just need to make the selected label 0 and then you are
effectively painting with the background color. Remember you can use the color
picker tool at any point to change the selected label.

You can adjust the size of your paintbrush using the brush size slider, making
it as small as a single pixel for incredibly detailed painting.

If you have a multidimensional labels layer then your paintbrush will only edit
data in the visible slice by default; however, if you enable the `n_dimensional`
property and paintbrush then your paintbrush will extend out into neighbouring
slices according to its size.

You can quickly select the paintbrush by pressing the `P` key when the labels
layer is selected.

## Using the fill bucket

Sometimes you might want to replace an entire label with a different label. This
could be because you want to make two touching regions have the same label, or
you want to just replace one label with a different one, or maybe you have
painted around the edge of a region and you want to quickly fill in its inside.
To do this you can select the `fill bucket` by clicking on the droplet icon in
the layer controls panel and then click on a target region of interest in the
layer. The fill bucket will fill using the currently selected label.

By default, the fill bucket will only change contiguous or connected pixels of
the same label as the pixel that is clicked on. If you want to change all the
pixels of that label regardless of where they are in the slice, then you can set
the `contiguous` property or checkbox to `False`.

If you have a multidimensional labels layer then your fill bucket will only edit
data in the visible slice by default; however, if you enable the `n_dimensional`
property and paintbrush then your fill bucket will extend out into neighbouring
slices, either to all pixels with that label in the layer, or only connected
pixels depending on if the contiguous property is disabled or not.

You can quickly select the fill bucket by pressing the `F` key when the labels
layer is selected.

## Creating, deleting, merging, and splitting connected components

Using the `color picker`, `paintbrush`, and `fill bucket` tools one can create
and edit object segmentation maps. Below we show how to use these tools to by
perform common editing tasks on connected components (keep the `contiguous` box
checked).

**Drawing a connected component**:

![image: draw component](../../images/draw_component.webm)

Press `M` to select a new label color. Select the `paintbrush` tool and draw a
closed contour around the object. Select the `fill bucket` tool and click inside
the contour to assign the label to all pixels of the object.

**Selecting a connected component**:

![image: delete label](../../images/delete_label.webm)

Select the background label with the `color picker` (alternative: press keyboard
shortcut `E`), then use the `fill bucket` to set all pixels of the connected
component to background.

**Merging connected components**:

![image: merge labels](../../images/merge_labels.webm)

Select the label of one of the components with the `color picker` tool and then
filling the components to be merged with the fill bucket.

**Splitting a connected component**:

![image: split label](../../images/split_label.webm)

Splitting a connected component will introduce an additional object, therefore
press `M` to select a label number that is not already in use. Use the
paintbrush tool to draw a dividing line, then assign the new label to one of the
parts with the `fill bucket`.

## Undo / redo functionality

When painting or using the fill bucket it can be easy to make a mistake that you
might want to undo or then redo. For the labels layer we support an undo with
`ctrl-Z` and redo with `shift-ctrl-Z`. We plan to support this sort of
functionality more generally, but for now these actions will undo the most
recent painting or filling event, up to 100 events in the past.

If you have multidimensional data, then adjusting the currently viewed slice
will cause the undo history to be reset.
