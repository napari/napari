from napari.utils.colormaps import simple_colormaps
import numpy as np
from napari.layers import Image


def stack_to_images(stack, axis):
    """Function to split the active layer into separate layers along an axis

    Parameters
    ----------
    stack : napari.layers.Image
        The image stack to be split into a list of image layers
    axis : int
        The axis to split along.

    Returns
    -------
    list
        List of images
    """

    if not isinstance(stack, Image):
        print("Active layer is not an image")
        return None

    data = stack.data
    name = stack.name

    cmaps = list(simple_colormaps.keys())
    num_dim = len(data.shape)

    if num_dim < 3:
        print("not enough dimensions")
        return None

    if axis >= num_dim:
        print("the image has {} dimensions".format(num_dim))
        return None

    imagelist = list()
    for i in range(data.shape[axis]):
        layer_name = "{:02d}_{}".format(i, name)

        try:
            color = cmaps[i]
        except IndexError:
            color = 'gray'

        image = Image(
            np.take(data, i, axis=axis),
            blending='additive',
            colormap=color,
            name=layer_name,
        )

        imagelist.append(image)

    return imagelist


def images_to_stack(images, axis=-1):
    """Function to combine selected image layers in one layer

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The viewer with the selected image
    axis : int
        Index to to insert the new axis

    Returns
    -------
    stack : napari.layers.Image
        Combined image stack
    """

    nc = len(images)
    if nc < 2:
        print("less than two layers selected")
        return None

    new_list = list()
    for i in range(nc):
        if not isinstance(images[i], Image):
            print("Selected layer {} is not an image".format(images[i].name))
            return None
        if i == 0:
            shape = images[i].data.shape
        else:
            if shape != images[i].data.shape:
                print("Layers not the same size")
                return None
        new_list.append(images[i].data)

    new_data = np.stack(new_list, axis=axis)
    stack = Image(new_data)

    return stack
