from napari.components.add_layers_mixin import AddLayersMixin
import warnings
from typing import Union, List, Tuple, Dict
import itertools

import numpy as np
from napari.layers import Image, Labels
from vispy.color import Colormap
from napari.utils.colormaps import colormaps
from napari.layers.image._image_utils import guess_multiscale
from napari.utils.misc import ensure_iterable, ensure_sequence_of_iterables


class StackUtils(AddLayersMixin):
    def __init__(self):
        super().__init__()

    def split_channels(
        self, data: np.ndarray, channel_axis: int, **kwargs,
    ) -> List[Tuple]:

        # Determine if data is a multiscale
        multiscale = kwargs['multiscale']
        if multiscale is None:
            multiscale, data = guess_multiscale(data)
        kwargs['multiscale'] = multiscale
        n_channels = (data[0] if multiscale else data).shape[channel_axis]
        kwargs['blending'] = kwargs['blending'] or 'additive'

        # these arguments are *already* iterables in the single-channel case.
        iterable_kwargs = {'scale', 'translate', 'contrast_limits', 'metadata'}

        # turn the kwargs dict into a mapping of {key: iterator}
        # so that we can use {k: next(v) for k, v in kwargs.items()} below
        for key, val in kwargs.items():
            if key == 'colormap' and val is None:
                if n_channels == 1:
                    kwargs[key] = iter(['gray'])
                elif n_channels == 2:
                    kwargs[key] = iter(colormaps.MAGENTA_GREEN)
                else:
                    kwargs[key] = itertools.cycle(colormaps.CYMRGB)

            # make sure that iterable_kwargs are a *sequence* of iterables
            # for the multichannel case.  For example: if scale == (1, 2) &
            # n_channels = 3, then scale should == [(1, 2), (1, 2), (1, 2)]
            elif key in iterable_kwargs:
                kwargs[key] = iter(
                    ensure_sequence_of_iterables(val, n_channels)
                )
            else:
                kwargs[key] = iter(ensure_iterable(val))

        layerdata_list = list()
        for i in range(n_channels):
            if multiscale:
                image = [
                    np.take(data[j], i, axis=channel_axis)
                    for j in range(len(data))
                ]
            else:
                image = np.take(data, i, axis=channel_axis)
            i_kwargs = {}
            for key, val in kwargs.items():
                try:
                    i_kwargs[key] = next(val)
                except StopIteration:
                    raise IndexError(
                        "Error adding multichannel image with data shape "
                        f"{data.shape!r}.\nRequested channel_axis "
                        f"({channel_axis}) had length {n_channels}, but "
                        f"the '{key}' argument only provided {i} values. "
                    )

            layerdata = (image, i_kwargs, 'image')
            layerdata_list.append(layerdata)

        return layerdata_list

    def stack_to_images(
        self, stack: Image, axis: int, **kwargs: Dict,
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

        data, meta, _ = stack.as_layer_data_tuple()

        for key in ("contrast_limits", "colormap", "blending"):
            meta[key] = None

        name = stack.name
        num_dim = len(data.shape)
        # n_channels = data.shape[axis]

        if num_dim < 3:
            warnings.warn(
                "The image needs more than 2 dimensions for splitting",
                UserWarning,
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

        if 'colormap' in kwargs and kwargs['colormap'] is not None:
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

        meta.update(kwargs)
        imagelist = list()
        layerdata_list = self.split_channels(data, axis, **meta)
        for i, tup in enumerate(layerdata_list):
            idata, imeta, _ = tup
            layer_name = f'{name} layer {i}'
            imeta['name'] = layer_name

            image = Image(idata, **imeta)

            imagelist.append(image)

        return imagelist

    def images_to_stack(
        self,
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


if __name__ == '__main__':
    x = np.random.randint(0, 255, (4, 128, 128))
    testI = Image(x)
    data, meta, _ = testI.as_layer_data_tuple()
    s = StackUtils()
    d = s.stack_to_images(testI, 0, colormap=['red', 'blue'])
    print(len(d))
