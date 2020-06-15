import warnings
from typing import Union, List
import itertools

import numpy as np
from napari.layers import Image, Labels
from vispy.color import Colormap
from ...utils import colormaps


def stack_to_images(
    stack: Union[Image, Labels],
    axis: int,
    colormap: List[Union[str, Colormap]] = None,
    blending: str = None,
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
    n_channels = data.shape[axis]

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

    if colormap is None:
        if n_channels == 2:
            colormap = iter(colormaps.MAGENTA_GREEN)
        if n_channels > 2:
            colormap = itertools.cycle(colormaps.CYMRGB)

    if blending not in ['additive', 'translucent', 'opaque']:
        blending = 'additive'

    imagelist = list()

    for i in range(data.shape[axis]):
        layer_name = f'{name} layer {i}'
        try:
            color = next(colormap)
        except IndexError:
            color = 'gray'

        image = Image(
            np.take(data, i, axis=axis),
            blending=blending,
            colormap=color,
            name=layer_name,
        )

        imagelist.append(image)

    return imagelist


def images_to_stack(
    images: List[Union[Image, Labels]], axis: int = 0
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

    new_list = [image.data for image in images]
    new_data = np.stack(new_list, axis=axis)
    stack = Image(new_data)

    return stack
