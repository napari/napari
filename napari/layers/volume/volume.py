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
from ...util.colormaps import COLORMAPS_3D_DATA

# from ._constants import Camera


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

    _colormaps = COLORMAPS_3D_DATA
    # default_camera = str(Camera.TURNTABLE)

    def __init__(
        self,
        volume,
        *,
        metadata=None,
        multichannel=None,
        colormap='fire',
        clim=None,
        clim_range=None,
        interpolation='nearest',
        name=None,
        **kwargs,
    ):

        visual = VolumeNode(
            volume,
            method='translucent',
            threshold=0.225,
            cmap=self._colormaps[colormap],
        )
        super().__init__(visual, name)

        self.translate = (0, 0, 0)
        self.scale = (0.009, 0.009, 0.009, 1)

        self.events.add(clim=Event, colormap=Event, camera=Event)

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

            # self.camera = default_camera

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

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
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        indices = (
            slice(None, None, None),
            slice(None, None, None),
            slice(None, None, None),
        )
        self._data_view = np.asarray(self.data[tuple(indices)])

        self._node.set_data(self._data_view)

        self._need_visual_update = True
        self._update()

        self._data_thumbnail = self._data_view

    def _update_thumbnail(self):
        pass

    # @property
    # def camera(self):
    #     """Camera: Camera mode.
    #         Selects a preset camera mode in vispy that determines how
    #         volume is displayed
    #         Camera.TURNTABLE
    #         Camera.FLY
    #         Camera.ARCBALL
    #     """
    #     return str(self.camera)

    # @camera.setter
    # def camera(self, camera):
    #     if isinstance(camera, str):
    #         camera = Camera(camera)

    #     self._camera = camera
    #     self._node.camera
    #     self._node.update()
    #     self.events.camera()
