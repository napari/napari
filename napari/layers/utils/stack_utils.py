import itertools
from typing import Dict, List

import numpy as np

from ...layers import Image
from ...layers.image._image_utils import guess_multiscale
from ...types import FullLayerData
from ...utils.colormaps import CYMRGB, MAGENTA_GREEN, Colormap
from ...utils.misc import ensure_iterable, ensure_sequence_of_iterables
from ...utils.translations import trans


def slice_from_axis(array, *, axis, element):
    """Take a single index slice from array using slicing.

    Equivalent to :func:`np.take`, but using slicing, which ensures that the
    output is a view of the original array.

    Parameters
    ----------
    array : NumPy or other array
        Input array to be sliced.
    axis : int
        The axis along which to slice.
    element : int
        The element along that axis to grab.

    Returns
    -------
    sliced : NumPy or other array
        The sliced output array, which has one less dimension than the input.
    """
    slices = [slice(None) for i in range(array.ndim)]
    slices[axis] = element
    return array[tuple(slices)]


def split_channels(
    data: np.ndarray,
    channel_axis: int,
    **kwargs,
) -> List[FullLayerData]:
    """Split the data array into separate arrays along an axis.

    Keyword arguments will override any parameters altered or set in this
    function. Colormap, blending, or multiscale are set as follows if not
    overridden by a keyword:
    - colormap : (magenta, green) for 2 channels, (CYMRGB) for more than 2
    - blending : additive
    - multiscale : determined by layers.image._image_utils.guess_multiscale.

    Colormap, blending and multiscale will be set and returned in meta if not in kwargs.
    If any other key is not present in kwargs it will not be returned in the meta
    dictionary of the returned LaterData tuple. For example, if gamma is not in
    kwargs then meta will not have a gamma key.

    Parameters
    ----------
    data : array or list of array
    channel_axis : int
        Axis to split the image along.
    **kwargs : dict
        Keyword arguments will override the default image meta keys
        returned in each layer data tuple.

    Returns
    -------
    List of LayerData tuples: [(data: array, meta: Dict, type: str )]
    """

    # Determine if data is a multiscale
    multiscale = kwargs.get('multiscale')
    if not multiscale:
        multiscale, data = guess_multiscale(data)
        kwargs['multiscale'] = multiscale

    n_channels = (data[0] if multiscale else data).shape[channel_axis]

    kwargs['blending'] = kwargs.get('blending') or 'additive'
    kwargs.setdefault('colormap', None)
    # these arguments are *already* iterables in the single-channel case.
    iterable_kwargs = {
        'scale',
        'translate',
        'affine',
        'contrast_limits',
        'metadata',
    }

    # turn the kwargs dict into a mapping of {key: iterator}
    # so that we can use {k: next(v) for k, v in kwargs.items()} below
    for key, val in kwargs.items():
        if key == 'colormap' and val is None:
            if n_channels == 1:
                kwargs[key] = iter(['gray'])
            elif n_channels == 2:
                kwargs[key] = iter(MAGENTA_GREEN)
            else:
                kwargs[key] = itertools.cycle(CYMRGB)

        # make sure that iterable_kwargs are a *sequence* of iterables
        # for the multichannel case.  For example: if scale == (1, 2) &
        # n_channels = 3, then scale should == [(1, 2), (1, 2), (1, 2)]
        elif key in iterable_kwargs or (
            key == 'colormap' and isinstance(val, Colormap)
        ):
            kwargs[key] = iter(ensure_sequence_of_iterables(val, n_channels))
        else:
            kwargs[key] = iter(ensure_iterable(val))

    layerdata_list = list()
    for i in range(n_channels):
        if multiscale:
            image = [
                slice_from_axis(data[j], axis=channel_axis, element=i)
                for j in range(len(data))
            ]
        else:
            image = slice_from_axis(data, axis=channel_axis, element=i)
        i_kwargs = {}
        for key, val in kwargs.items():
            try:
                i_kwargs[key] = next(val)
            except StopIteration:
                raise IndexError(
                    trans._(
                        "Error adding multichannel image with data shape {data_shape!r}.\nRequested channel_axis ({channel_axis}) had length {n_channels}, but the '{key}' argument only provided {i} values. ",
                        deferred=True,
                        data_shape=data.shape,
                        channel_axis=channel_axis,
                        n_channels=n_channels,
                        key=key,
                        i=i,
                    )
                )

        layerdata = (image, i_kwargs, 'image')
        layerdata_list.append(layerdata)

    return layerdata_list


def stack_to_images(
    stack: Image,
    axis: int,
    **kwargs: Dict,
) -> List[Image]:
    """Splits a single Image layer into a list layers along axis.

    Some image layer properties will be changed unless specified as an item in
    kwargs. Properties such as colormap and contrast_limits are set on individual
    channels. Properties will be changed as follows (unless overridden with a kwarg):
    - colormap : (magenta, green) for 2 channels, (CYMRGB) for more than 2
    - blending : additive
    - contrast_limits : min and max of the image

    All other properties, such as scale and translate will be propagated from the
    original stack, unless a keyword argument passed for that property.

    Parameters
    ----------
    stack : napari.layers.Image
        The image stack to be split into a list of image layers
    axis : int
        The axis to split along.

    Returns
    -------
    imagelist: list
        List of Image objects
    """

    data, meta, _ = stack.as_layer_data_tuple()

    for key in ("contrast_limits", "colormap", "blending"):
        del meta[key]

    name = stack.name
    num_dim = stack.ndim

    if num_dim < 3:
        raise ValueError(
            trans._(
                "The image needs more than 2 dimensions for splitting",
                deferred=True,
            )
        )

    if axis >= num_dim:
        raise ValueError(
            trans._(
                "Can't split along axis {axis}. The image has {num_dim} dimensions",
                deferred=True,
                axis=axis,
                num_dim=num_dim,
            )
        )

    if kwargs.get("colormap"):
        kwargs['colormap'] = itertools.cycle(kwargs['colormap'])

    if meta['rgb']:
        if axis == (num_dim - 1) or axis == -1:
            kwargs['rgb'] = False  # split channels as grayscale
        else:
            kwargs['rgb'] = True  # split some other axis, remain rgb
            meta['scale'].pop(axis)
            meta['translate'].pop(axis)
    else:
        kwargs['rgb'] = False
        meta['scale'].pop(axis)
        meta['translate'].pop(axis)

    meta['rotate'] = None
    meta['shear'] = None
    meta['affine'] = None

    meta.update(kwargs)
    imagelist = list()
    layerdata_list = split_channels(data, axis, **meta)
    for i, tup in enumerate(layerdata_list):
        idata, imeta, _ = tup
        layer_name = f'{name} layer {i}'
        imeta['name'] = layer_name

        imagelist.append(Image(idata, **imeta))

    return imagelist


def images_to_stack(
    images: List[Image],
    axis: int = 0,
    **kwargs: Dict,
) -> Image:
    """Combines a list of Image layers into one layer stacked along axis

    The new image layer will get the meta properties of the first
    image layer in the input list unless specified in kwargs

    Parameters
    ----------
    images : List
        List of Image Layers
    axis : int
        Index to to insert the new axis
    **kwargs : dict
        Dictionary of parameters values to override parameters
        from the first image in images list.

    Returns
    -------
    stack : napari.layers.Image
        Combined image stack
    """

    if len(images) == 0:
        raise IndexError(
            trans._(
                "images list is empty",
                deferred=True,
            )
        )

    data, meta, _ = images[0].as_layer_data_tuple()

    kwargs.setdefault("scale", np.insert(meta['scale'], axis, 1))
    kwargs.setdefault("translate", np.insert(meta['translate'], axis, 0))

    meta.update(kwargs)
    new_data = np.stack([image.data for image in images], axis=axis)
    stack = Image(new_data, **meta)

    return stack
