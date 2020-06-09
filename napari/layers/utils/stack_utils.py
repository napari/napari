import warnings
from typing import Union, List

import numpy as np
from napari.layers import Image, Labels
from vispy.color import Colormap


def stack_to_images(
    stack: Union[Image, Labels],
    axis: int,
    colormaps: List[Union[str, Colormap]] = None,
) -> List[Image]:
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

    data = stack.data
    name = stack.name
    num_dim = len(data.shape)

    if num_dim < 3:
        warnings.warn(
            "The image needs more than 2 dimensions for splitting", UserWarning
        )
        return None

    if axis >= num_dim:
        warnings.warn(
            "Can't split along axis {}. The image has {} dimensions".format(
                axis, num_dim
            ),
            UserWarning,
        )
        return None

    if colormaps is None:
        colormaps = data.shape[axis] * ['gray']

    imagelist = list()

    for i in range(data.shape[axis]):
        layer_name = "{:02d}_{}".format(i, name)

        try:
            color = colormaps[i]
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


def images_to_stack(
    images: List[Union[Image, Labels]], axis: int = -1
) -> Image:
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
    new_list = list()
    for image in images:
        new_list.append(image.data)

    try:
        new_data = np.stack(new_list, axis=axis)
    except ValueError as err:
        warnings.warn("{}".format(err))
        return

    stack = Image(new_data)

    return stack
