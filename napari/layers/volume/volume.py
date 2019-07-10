import warnings
import numpy as np
from copy import copy
from scipy import ndimage as ndi
import vispy.color
from ..base import Layer
from ..._vispy.scene.visuals import Volume as VolumeNode
from ...util.misc import (
    is_multichannel,
    calc_data_range,
    increment_unnamed_colormap,
)
from ...util.event import Event
from ...util.colormaps import AVAILABLE_COLORMAPS


class Volume(Layer):
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

    Extended Summary
    ----------
    _data_view : array (N, M, K), (N, M, K, 3), or (N, M, K, 4)
        Volume data for the currently viewed slice. Must be 3D volume data, but
        can be multidimensional for RGB or RGBA volumes if multidimensional is
        `True`.
    """

    _colormaps = AVAILABLE_COLORMAPS

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

        visual = VolumeNode(volume, method='translucent')
        super().__init__(visual, name)

        self.events.add(clim=Event, colormap=Event)

        with self.freeze_refresh():
            # Set data
            self._data = volume
            self.metadata = metadata or {}
            self.multichannel = multichannel

            # Intitialize volume views and thumbnails with zeros
            if self.multichannel:
                self._data_view = np.zeros((1, 1, 1) + (self.shape[-1],))
            else:
                self._data_view = np.zeros((1, 1, 1))
            self._data_thumbnail = self._data_view

            # Set colormap
            self._colormap_name = ''
            self.colormap = colormap

            # Set update flags
            self._need_display_update = False
            self._need_visual_update = False

            # Re intitialize indices
            self._indices = (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )

            self._position = (0, 0, 0)
            self.coordinates = (0, 0, 0)
            self._thumbnail_shape = (32, 32, 32, 4)
            self._thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

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
                colormapped[..., 3] = downsampled[..., 3] * self.opacity
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
            colormapped[..., 3] *= self.opacity
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
        print(self.coordinates)
        coord[-3:] = np.clip(coord[-3:], 0, np.asarray(shape) - 1)

        value = self._data_view[tuple(coord[-3:])]

        return coord, value

    def get_message(self, coord, value):
        """Generate a status message based on the coordinates and information
        about what shapes are hovered over

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        value : int or float or sequence of int or float
            Value of the data at the coord.

        Returns
        ----------
        msg : string
            String containing a message that can be used as a status update.
        """

        msg = f'{coord}, {self.name}' + ', value '
        if isinstance(value, np.ndarray):
            if isinstance(value[0], np.integer):
                msg = msg + str(value)
            else:
                v_str = '[' + str.join(', ', [f'{v:0.3}' for v in value]) + ']'
                msg = msg + v_str
        else:
            if isinstance(value, np.integer):
                msg = msg + str(value)
            else:
                msg = msg + f'{value:0.3}'

        return msg

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if self.multichannel:
            self._multichannel = is_multichannel(data.shape)
        self.events.data()
        self.refresh()

    def _get_shape(self):
        if self.multichannel:
            return self.data.shape[:-1]
        return self.data.shape

    @property
    def multichannel(self):
        """bool: Whether the image is multichannel."""
        return self._multichannel

    @multichannel.setter
    def multichannel(self, multichannel):
        if multichannel is False:
            self._multichannel = multichannel
        else:
            # If multichannel is True or None then guess if multichannel
            # allowed or not, and if allowed set it to be True
            self._multichannel = is_multichannel(self.data.shape)
        self.refresh()

    @property
    def colormap(self):
        """2-tuple of str, vispy.color.Colormap: colormap for luminance images.
        """
        return self._colormap_name, self._node.cmap

    @colormap.setter
    def colormap(self, colormap):
        name = '[unnamed colormap]'
        if isinstance(colormap, str):
            name = colormap
        elif isinstance(colormap, tuple):
            name, cmap = colormap
            self._colormaps[name] = cmap
        elif isinstance(colormap, dict):
            self._colormaps.update(colormap)
            name = list(colormap)[0]  # first key in dict
        elif isinstance(colormap, vispy.color.Colormap):
            name = increment_unnamed_colormap(
                name, list(self._colormaps.keys())
            )
            self._colormaps[name] = colormap
        else:
            warnings.warn(f'invalid value for colormap: {colormap}')
            name = self._colormap_name
        self._colormap_name = name
        self._node.cmap = self._colormaps[name]
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    # wrap visual properties:
    @property
    def clim(self):
        """list of float: Limits to use for the colormap."""
        return list(self._node.clim)

    @clim.setter
    def clim(self, clim):
        self._clim_msg = f'{float(clim[0]): 0.3}, {float(clim[1]): 0.3}'
        self.status = self._clim_msg
        self._node.clim = clim
        if clim[0] < self._clim_range[0]:
            self._clim_range[0] = copy(clim[0])
        if clim[1] > self._clim_range[1]:
            self._clim_range[1] = copy(clim[1])
        self._update_thumbnail()
        self.events.clim()

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        indices = list(self.indices)
        indices[:-3] = np.clip(
            indices[:-3], 0, np.subtract(self.shape[:-2], 1)
        )
        self._data_view = np.asarray(self.data[tuple(indices)])

        self._node.set_data(self._data_view)

        self._need_visual_update = True
        self._update()

        coord, value = self.get_value()
        self.status = self.get_message(coord, value)

        self._data_thumbnail = self._data_view
        self._update_thumbnail()

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord, value = self.get_value()
        self.status = self.get_message(coord, value)
