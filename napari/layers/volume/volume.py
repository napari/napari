import warnings
import numpy as np
from copy import copy
from scipy import ndimage as ndi
import vispy.color
from vispy.scene.visuals import Volume as VolumeNode
from ..base import Layer
from ...util.misc import calc_data_range, increment_unnamed_colormap
from ...util.event import Event
from ...util.colormaps import AVAILABLE_COLORMAPS
from ._constants import Rendering


class Volume(Layer):
    """Volume layer.

    Parameters
    ----------
    volume : array
        Volumetric data, must be at least 3-dimensional.
    metadata : dict, optional
        Volume metadata.
    colormap : str, vispy.Color.Colormap, tuple, dict, keyword-only
        Colormap to use for luminance volumes. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    clim : list (2,), keyword-only
        Color limits to be used for determining the colormap bounds for
        luminance volumes. If not passed is calculated as the min and max
        of the volume.
    clim_range : list (2,), keyword-only
        Range for the color limits. If not passed is be calculated as the
        min and max of the volume. Passing a value prevents this
        calculation which can be useful when working with very larg
        datasets that are dynamically loaded.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    name : str, keyword-only
        Name of the layer.

    Attributes
    ----------
    data : array
        Volumetric data, must be at least 3-dimensional.
    metadata : dict
        Volume metadata.
    colormap : 2-tuple of str, vispy.color.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance volumes.
    colormaps : tuple of str
        Names of the available colormaps.
    clim : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance volumes.
    clim_range : list (2,) of float
        Range for the color limits for luminace volumes.

    Extended Summary
    ----------
    _data_view : array (N, M, K)
        Volume data for the currently viewed slice, must be 3-dimensional
    """

    class_keymap = {}
    _colormaps = AVAILABLE_COLORMAPS
    _default_rendering = Rendering.MIP.value

    def __init__(
        self,
        volume,
        *,
        metadata=None,
        colormap='gray',
        clim=None,
        clim_range=None,
        opacity=1,
        blending='translucent',
        visible=True,
        name=None,
        **kwargs,
    ):

        visual = VolumeNode(np.empty((1, 1, 1)))
        super().__init__(
            visual,
            name=name,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self._rendering = self._default_rendering

        self.events.add(clim=Event, colormap=Event, rendering=Event)

        with self.freeze_refresh():
            # Set data
            self._data = volume
            self.metadata = metadata or {}

            # Intitialize volume views and thumbnails with zeros
            self._data_view = np.zeros((1, 1, 1))
            self._data_thumbnail = self._data_view

            # Set clims and colormaps
            self._colormap_name = ''
            self._clim_msg = ''
            if clim_range is None:
                self._clim_range = calc_data_range(self.data)
            else:
                self._clim_range = clim_range

            if clim is None:
                self.clim = self._clim_range
            else:
                self.clim = clim
            self.colormap = colormap

            # Set update flags
            self._need_display_update = False
            self._need_visual_update = False

            # Re intitialize indices
            self._indices = (0,) * (self.ndim - 3) + (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.events.data()
        self.refresh()

    def _get_shape(self):
        return self.data.shape

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
        cmap = self._colormaps[name]
        self._node.view_program['texture2D_LUT'] = (
            cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
        )
        self._node.cmap = cmap
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    @property
    def clim(self):
        """list of float: Limits to use for the colormap."""
        return list(self._clim)

    @clim.setter
    def clim(self, clim):
        self._clim = clim
        self._clim_msg = f'{float(clim[0]): 0.3}, {float(clim[1]): 0.3}'
        self.status = self._clim_msg
        if clim[0] < self._clim_range[0]:
            self._clim_range[0] = copy(clim[0])
        if clim[1] > self._clim_range[1]:
            self._clim_range[1] = copy(clim[1])
        self.refresh()
        self.events.clim()

    @property
    def rendering(self):
        """Rendering: Rendering mode.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        if isinstance(rendering, str):
            rendering = Rendering(rendering)

        self._node.method = rendering.value
        self._rendering = rendering
        self.refresh()
        self.events.rendering()

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        indices = list(self.indices)
        indices[:-3] = np.clip(
            indices[:-3], 0, np.subtract(self.shape[:-3], 1)
        )
        self._data_view = np.asarray(self.data[tuple(indices)])

        self._node.set_data(self._data_view, clim=self.clim)

        self._need_visual_update = True
        self._update()

        self._data_thumbnail = self._data_view
        self._update_thumbnail()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        # take max projection of volume along first axis
        image = np.max(self._data_thumbnail, axis=0)
        # print(image.shape)
        zoom_factor = np.divide(
            self._thumbnail_shape[:2], image.shape[:2]
        ).min()
        # warning filter can be removed with scipy 1.4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downsampled = ndi.zoom(
                image, zoom_factor, prefilter=False, order=0
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
