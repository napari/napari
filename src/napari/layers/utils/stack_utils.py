from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pint

from napari.layers import Image
from napari.layers.image._image_utils import guess_multiscale
from napari.utils.colormaps import CMYBGR, MAGENTA_GREEN, Colormap
from napari.utils.misc import ensure_iterable, ensure_sequence_of_iterables
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.types import FullLayerData


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
    # Check if array is a zarr array and wrap it with dask
    # to keep lazy behavior and avoid loading into memory
    if hasattr(array, '__module__') and array.__module__.startswith('zarr'):
        import dask.array as da

        array = da.from_zarr(array)
        warnings.warn(
            trans._(
                'zarr array cannot be sliced lazily, converted to dask array.',
                deferred=True,
            )
        )

    slices = [slice(None) for i in range(array.ndim)]
    slices[axis] = element
    return array[tuple(slices)]


def split_channels(
    data: np.ndarray,
    channel_axis: int,
    **kwargs,
) -> list[FullLayerData]:
    """Split the data array into separate arrays along an axis.

    Keyword arguments will override any parameters altered or set in this
    function. Colormap, blending, or multiscale are set as follows if not
    overridden by a keyword:
    - colormap : (magenta, green) for 2 channels, (CMYBGR) for more than 2
    - blending : translucent for first channel, additive for others
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
    # Use original blending mode or for multichannel use translucent for first channel then additive
    kwargs['blending'] = kwargs.get('blending') or ['translucent_no_depth'] + [
        'additive'
    ] * (n_channels - 1)
    kwargs.setdefault('colormap', None)
    # these arguments are *already* iterables in the single-channel case.
    iterable_kwargs = {
        'axis_labels',
        'scale',
        'translate',
        'contrast_limits',
        'metadata',
        'plane',
        'experimental_clipping_planes',
        'custom_interpolation_kernel_2d',
        'units',
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
                kwargs[key] = itertools.cycle(CMYBGR)

        # make sure that iterable_kwargs are a *sequence* of iterables
        # for the multichannel case.  For example: if scale == (1, 2) &
        # n_channels = 3, then scale should == [(1, 2), (1, 2), (1, 2)]
        elif key in iterable_kwargs or (
            key == 'colormap' and isinstance(val, Colormap)
        ):
            kwargs[key] = iter(
                ensure_sequence_of_iterables(
                    val,
                    n_channels,
                    repeat_empty=True,
                    allow_none=True,
                )
            )
        elif key in ['rotate', 'shear'] or (
            key == 'affine' and isinstance(val, np.ndarray)
        ):
            # affine may be Affine or np.ndarray object that is not
            # iterable, but it is not now a problem as we use it only to warning
            # if a provided object is a sequence and channel_axis is not provided
            kwargs[key] = itertools.repeat(val, n_channels)
        else:
            kwargs[key] = iter(ensure_iterable(val))

    layerdata_list = []
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
            except StopIteration as e:
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
                ) from e

        layerdata: FullLayerData = (image, i_kwargs, 'image')
        layerdata_list.append(layerdata)

    return layerdata_list


def stack_to_images(stack: Image, axis: int, **kwargs) -> list[Image]:
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

    for key in ('contrast_limits', 'colormap', 'blending'):
        del meta[key]

    name = stack.name
    num_dim = 3 if stack.rgb else stack.ndim

    if num_dim < 3:
        raise ValueError(
            trans._(
                'The image needs more than 2 dimensions for splitting',
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

    if kwargs.get('colormap'):
        kwargs['colormap'] = itertools.cycle(kwargs['colormap'])

    if meta['rgb']:
        if axis in [num_dim - 1, -1]:
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
    meta['axis_labels'] = None
    meta['units'] = None

    meta.update(kwargs)
    imagelist = []
    layerdata_list = split_channels(data, axis, **meta)
    for i, tup in enumerate(layerdata_list):
        idata, imeta, _ = tup
        layer_name = f'{name} layer {i}'
        imeta['name'] = layer_name

        imagelist.append(Image(idata, **imeta))

    return imagelist


def split_rgb(stack: Image, with_alpha=False) -> list[Image]:
    """Split RGB image into separate channel images while preserving affine transforms."""
    if not stack.rgb:
        raise ValueError(
            trans._('Image must be RGB to use split_rgb', deferred=True)
        )

    data, meta, _ = stack.as_layer_data_tuple()

    meta['colormap'] = ('red', 'green', 'blue', 'gray')
    meta['rgb'] = False

    layerdata_list = split_channels(data, channel_axis=-1, **meta)

    images = [
        Image(image, **i_kwargs) for image, i_kwargs, _ in layerdata_list
    ]

    # first (red) channel blending is inherited from RGB stack
    # green and blue are set to additive to maintain appearance of unsplit RGB
    # if rgba, set alpha channel blending to multiplicative
    for img in images[1:]:
        img.blending = 'additive'
    if with_alpha:
        images[-1].blending = 'multiplicative'

    return images if with_alpha else images[:3]


def images_to_stack(images: list[Image], axis: int = 0, **kwargs) -> Image:
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

    if not images:
        raise IndexError(trans._('images list is empty', deferred=True))

    if not all(isinstance(layer, Image) for layer in images):
        non_image_layers = [
            (layer.name, type(layer).__name__)
            for layer in images
            if not isinstance(layer, Image)
        ]
        raise ValueError(
            trans._(
                'All selected layers to be merged must be Image layers. '
                'The following layers are not Image layers: '
                f'{", ".join(f"{name} ({layer_type})" for name, layer_type in non_image_layers)}'
            )
        )

    data, meta, _ = images[0].as_layer_data_tuple()

    # RGB images do not need extra dimensions inserted into metadata
    if 'rgb' not in kwargs:
        kwargs.setdefault('scale', np.insert(meta['scale'], axis, 1))
        kwargs.setdefault('translate', np.insert(meta['translate'], axis, 0))

    meta.update(kwargs)

    # Check if input images are either all multiscale or not
    multiscale_flags = [
        getattr(image, 'multiscale', False) for image in images
    ]
    if not all(multiscale_flags) and any(multiscale_flags):
        raise ValueError(
            trans._(
                'All images must have the same multiscale status (all True or all False) to be stacked.\nGot: {multiscale_flags}',
                multiscale_flags=multiscale_flags[::-1],
                deferred=True,
            )
        )

    if all(multiscale_flags):
        # Check that all multiscale images have the same number of levels
        n_scales_list = [len(image.data) for image in images]
        if len(set(n_scales_list)) != 1:
            raise ValueError(
                trans._(
                    'All multiscale images must have the same number of levels to be stacked.\nGot: {n_scales_list}',
                    deferred=True,
                    n_scales_list=n_scales_list,
                )
            )
        arrays_to_check = [img.data[0] for img in images]
    else:
        arrays_to_check = [img.data for img in images]

    # check if any of the data arrays are zarr arrays
    # zarr doesn't have a stack method, so we will use dask.array.stack
    is_zarr = any(
        hasattr(arr, '__module__') and arr.__module__.startswith('zarr')
        for arr in arrays_to_check
    )

    if is_zarr:
        import dask.array as da

        stacker = da.stack
        warnings.warn(
            trans._(
                'zarr array cannot be stacked lazily, using dask array to stack.',
                deferred=True,
            )
        )
    else:
        stacker = np.stack

    if all(multiscale_flags):
        n_scales = len(images[0].data)
        new_data = [
            stacker([image.data[level] for image in images], axis=axis)
            for level in range(n_scales)
        ]
    else:
        new_data = stacker([image.data for image in images], axis=axis)

    # RGB images do not need extra dimensions inserted into metadata
    # They can use the meta dict from one of the source image layers
    if not meta['rgb']:
        meta['units'] = (pint.get_application_registry().pixel,) + meta[
            'units'
        ]
        meta['axis_labels'] = (f'axis -{data.ndim + 1}',) + meta['axis_labels']

    return Image(new_data, **meta)


def merge_rgb(images: list[Image]) -> Image:
    """Variant of images_to_stack that makes an RGB from 3 images."""
    if not (
        len(images) in [3, 4] and all(isinstance(x, Image) for x in images)
    ):
        raise ValueError(
            trans._(
                'Merging to RGB requires either 3 or 4 Image layers',
                deferred=True,
            )
        )
    if not all(image.data.shape == images[0].data.shape for image in images):
        all_shapes = [(image.name, image.data.shape) for image in images]
        raise ValueError(
            trans._(
                'Shape mismatch! To merge to RGB, all selected Image layers (with R, G, and B colormaps) must have the same shape. '
                'Mismatched shapes: '
                f'{", ".join(f"{name} (shape: {shape})" for name, shape in all_shapes)}'
            )
        )

    # we will check for the presence of R G B colormaps to determine how to merge
    colormaps = {image.colormap.name for image in images}
    r_g_b = ['red', 'green', 'blue']
    # if image is rgba, add gray colormap to represent alpha channel
    if len(colormaps) == 4:
        r_g_b.append('gray')
    if colormaps != set(r_g_b):
        missing_colormaps = set(r_g_b) - colormaps
        raise ValueError(
            trans._(
                'Missing colormap(s): {missing_colormaps}! To merge layers to '
                f'{"RGB" if len(r_g_b) == 3 else "RGBA"}, ensure you have '
                f'{", ".join(r_g_b[:-1])}, and {r_g_b[-1]} as layer colormaps.',
                missing_colormaps=missing_colormaps,
                deferred=True,
            )
        )

    # use the R G B colormaps to order the images for merging
    imgs = [
        image
        for color in r_g_b
        for image in images
        if image.colormap.name == color
    ]
    return images_to_stack(imgs, axis=-1, rgb=True)
