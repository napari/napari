import warnings
from typing import Union, List, Tuple
import itertools

import numpy as np
from napari.layers import Image, Labels
from vispy.color import Colormap
from ...utils import colormaps


def stack_to_images(
    stack: Image,
    axis: int,
    colormap: List[Union[str, Colormap]] = None,
    contrast_limits: List[List[int]] = None,
    gamma: List[float] = None,
    blending: str = None,
    scale: Tuple[float] = None,
    translate: Tuple[float] = None,
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
        elif n_channels > 2:
            colormap = itertools.cycle(colormaps.CYMRGB)
        else:
            colormap = itertools.repeat('gray')
    else:
        colormap = itertools.cycle(colormap)

    if blending not in ['additive', 'translucent', 'opaque']:
        blending = 'additive'

    if scale is None:
        if stack.rgb:
            if axis == (num_dim - 1) or axis == -1:
                scale = stack.scale
            else:
                scale = np.delete(stack.scale, axis)
        else:
            scale = np.delete(stack.scale, axis)

    if translate is None:
        if stack.rgb:
            if axis == (num_dim - 1) or axis == -1:
                translate = stack.translate
            else:
                translate = np.delete(stack.translate, axis)
        else:
            translate = np.delete(stack.translate, axis)

    if contrast_limits is None:
        contrast_limits = n_channels * [None]

    if gamma is None:
        gamma = n_channels * [1]

    if stack.rgb:
        if axis == (num_dim - 1) or axis == -1:
            rgb = False  # split channels as grayscale
        else:
            rgb = True  # split some other axis, remain rgb
    else:
        rgb = False

    kwargs = {
        'rgb': rgb,
        'blending': blending,
        'interpolation': stack.interpolation,
        'rendering': stack.rendering,
        'iso_threshold': stack.iso_threshold,
        'attenuation': stack.attenuation,
        'metadata': stack.metadata,
        'scale': scale,
        'translate': translate,
        'opacity': stack.opacity,
        'visible': stack.visible,
        'multiscale': stack.multiscale,
    }

    imagelist = list()

    for i in range(n_channels):
        layer_name = f'{name} layer {i}'

        kwargs['contrast_limits'] = contrast_limits[i]
        kwargs['gamma'] = gamma[i]
        kwargs['colormap'] = next(colormap)

        image = Image(np.take(data, i, axis=axis), name=layer_name, **kwargs)

        imagelist.append(image)

    return imagelist


def images_to_stack(
    images: List[Union[Image, Labels]],
    axis: int = 0,
    rgb: bool = None,
    colormap: Union[str, Colormap] = 'gray',
    contrast_limits: List[int] = None,
    gamma: float = 1,
    interpolation: str = 'nearest',
    rendering: str = 'mip',
    iso_threshold: float = 0.5,
    attenuation: float = 0.5,
    name: str = None,
    metadata: dict = None,
    scale: Tuple[float] = None,
    translate: Tuple[float] = None,
    opacity: float = 1,
    blending: str = 'translucent',
    visible: bool = True,
    multiscale: bool = None,
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

    print(scale, translate)
    print(type(images[0].scale), type(images[0].translate))
    if scale is None:
        scale = np.insert(images[0].scale, axis, 1)
    if translate is None:
        translate = np.insert(images[0].translate, axis, 0)

    print(scale, translate)
    kwargs = {
        'rgb': rgb,
        'colormap': colormap,
        'blending': blending,
        'contrast_limits': contrast_limits,
        'gamma': gamma,
        'interpolation': interpolation,
        'rendering': rendering,
        'iso_threshold': iso_threshold,
        'attenuation': attenuation,
        'name': name,
        'metadata': metadata,
        'scale': scale,
        'translate': translate,
        'opacity': opacity,
        'visible': visible,
        'multiscale': multiscale,
    }

    new_list = [image.data for image in images]
    new_data = np.stack(new_list, axis=axis)
    stack = Image(new_data, **kwargs)

    return stack
