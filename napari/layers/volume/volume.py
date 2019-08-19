import warnings
import numpy as np
from copy import copy
from scipy import ndimage as ndi
import vispy.color
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
    scale : list, optional
        List of anisotropy factors to scale the volume by. Must be one for
        each dimension.
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
    scale : list
        List of anisotropy factors to scale the volume by. Must be one for
        each dimension.

    Extended Summary
    ----------
    _data_view : array (N, M, K)
        Volume data for the currently viewed slice, must be 3-dimensional
    """

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
        scale=None,
        name=None,
        **kwargs,
    ):

        super().__init__(
            name=name, opacity=opacity, blending=blending, visible=visible
        )

        self._rendering = self._default_rendering

        self.events.add(clim=Event, colormap=Event, rendering=Event)

        # Set data
        self._data = volume
        self.metadata = metadata or {}

        self.dims.ndim = volume.ndim
        with self.dims.events.display.blocker():
            self.dims.ndisplay = 3

        self._scale = scale or [1] * self.dims.ndim
        self._translate = [0] * self.dims.ndim
        self._position = (0,) * self.dims.ndim

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

        # Trigger generation of view slice and thumbnail
        self._update_dims()
        self._set_view_slice()

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._update_dims()
        self._set_view_slice()
        self.events.data()

    def _get_range(self):
        return tuple(
            (0, m, 1) for m in np.multiply(self.data.shape, self.scale)
        )

    @property
    def colormap(self):
        """2-tuple of str, vispy.color.Colormap: colormap for luminance images.
        """
        return self._colormap_name, self._cmap

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
        self._cmap = cmap
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

        self._rendering = rendering
        self.events.rendering()

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        self._data_view = np.asarray(self.data[self.dims.indices]).transpose(
            self.dims.displayed_order
        )

        self._data_thumbnail = self._data_view
        self._update_thumbnail()
        self._update_coordinates()
        self.events.set_data()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        # take max projection of volume along first axis
        image = np.max(self._data_thumbnail, axis=0)
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

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        value : int, float, or sequence of int or float, or None
            Value of the data at the coord, or none if coord is outside range.
        """
        coord = np.round(self.coordinates).astype(int)
        shape = self._data_view.shape

        if all(0 <= c < s for c, s in zip(coord[self.dims.displayed], shape)):
            value = self._data_view[tuple(coord[self.dims.displayed])]
        else:
            value = None

        return value
