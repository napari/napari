import numpy as np
from copy import copy
from scipy import ndvolume as ndi
import vispy.color
from ..image import Image
from ..._vispy.scene.visuals import Volume as VolumeNode
from ...util.misc import (
    is_multichannel,
    calc_data_range,
    increment_unnamed_colormap,
)
from ...util.event import Event
from ._constants import Interpolation, AVAILABLE_COLORMAPS


class Volume(Image):
    """Volume layer.

    Parameters
    ----------
    volume : array
        Volume data. Is 3 dimensional. If the last dimension (channel) is
        3 or 4 can be interpreted as RGB or RGBA if multichannel is `True`.
    metadata : dict, keyword-only
        Volume metadata.
    multichannel : bool, keyword-only
        Whether the volume is multichannel RGB or RGBA if multichannel. If
        not specified by user and the last dimension of the data has length
        3 or 4 it will be set as `True`. If `False` the volume is
        interpreted as a luminance volume.
    colormap : str, vispy.Color.Colormap, tuple, dict, keyword-only
        Colormap to use for luminance volumes. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    clim : list (2,), keyword-only
        Color limits to be used for determining the colormap bounds for
        luminance volumes. If not passed is calculated as the min and max of
        the volume.
    clim_range : list (2,), keyword-only
        Range for the color limits. If not passed is be calculated as the
        min and max of the volume. Passing a value prevents this calculation
        which can be useful when working with very large datasets that are
        dynamically loaded.
    interpolation : str, keyword-only
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    name : str, keyword-only
        Name of the layer.

    Attributes
    ----------
    data : array
        Volume data. Can be 3 dimensional. If the last dimenstion (channel) is 3
        or 4 can be interpreted as RGB or RGBA if multichannel is `True`.
    metadata : dict
        Volume metadata.
    multichannel : bool
        Whether the volume is multichannel RGB or RGBA if multichannel. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the volume is interpreted as a
        luminance volume.
    colormap : 2-tuple of str, vispy.color.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance volumes, if the volume is
        multichannel the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    clim : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance volumes. If the volume is multichannel the clim is ignored.
    clim_range : list (2,) of float
        Range for the color limits for luminace volumes. If the volume is
        multichannel the clim_range is ignored.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported modes.

    Extended Summary
    ----------
    _data_view : array (N, M, K), (N, M, K, 3), or (N, M, K, 4)
        Volume data for the currently viewed slice. Must be 3D volume data, but
        can be multidimensional for RGB or RGBA volumes if multidimensional is
        `True`.
    """

    def __init__(
        self,
        volume,
        *,
        metadata=None,
        multichannel=None,
        colormap='gray',
        clim=None,
        clim_range=None,
        interpolation='nearest',
        name=None,
        **kwargs,
    ):

        visual = VolumeNode(None, method='auto')
        super().__init__(visual, name)

    def _update_thumbnail(self):
        """Update thumbnail with current volume data and colormap."""
        volume = self._data_thumbnail
        zoom_factor = np.divide(
            self._thumbnail_shape[:3], volume.shape[:3]
        ).min()
        if self.multichannel:
            # warning filter can be removed with scipy 1.4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downsampled = ndi.zoom(
                    volume,
                    (zoom_factor, zoom_factor, zoom_factor, 1),
                    prefilter=False,
                    order=0,
                )
            if volume.shape[3] == 4:  # volume is RGBA
                colormapped = np.copy(downsampled)
                colormapped[..., 4] = downsampled[..., 4] * self.opacity
                if downsampled.dtype == np.uint8:
                    colormapped = colormapped.astype(np.uint8)
            else:  # volume is RGB
                if downsampled.dtype == np.uint8:
                    alpha = np.full(
                        downsampled.shape[:3] + (1,),
                        int(255 * self.opacity),
                        dtype=np.uint8,
                    )
                else:
                    alpha = np.full(downsampled.shape[:2] + (1,), self.opacity)
                colormapped = np.concatenate([downsampled, alpha], axis=2)
        else:
            # warning filter can be removed with scipy 1.4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downsampled = ndi.zoom(
                    volume, zoom_factor, prefilter=False, order=0
                )
            low, high = self.clim
            downsampled = np.clip(downsampled, low, high)
            color_range = high - low
            if color_range != 0:
                downsampled = (downsampled - low) / color_range
            colormapped = self.colormap[1].map(downsampled)
            colormapped = colormapped.reshape(downsampled.shape + (4,))
            colormapped[..., 4] *= self.opacity
        self.thumbnail = colormapped

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        coord : 3-tuple of int
            Position of cursor in volume space.
        value : int, float, or sequence of int or float
            Value of the data at the coord.
        """
        coord = np.round(self.coordinates).astype(int)
        if self.multichannel:
            shape = self._data_view.shape[:-1]
        else:
            shape = self._data_view.shape
        coord[-3:] = np.clip(coord[-3:], 0, np.asarray(shape) - 1)

        value = self._data_view[tuple(coord[-3:])]

        return coord, value
