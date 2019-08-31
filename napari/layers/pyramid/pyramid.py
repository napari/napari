import numpy as np
from ..image import Image


class Pyramid(Image):
    """Image pyramid layer.

    Parameters
    ----------
    data : list
        Pyramid data. List of array like image date. Each image can be N
        dimensional. If the last dimensions of the images have length 3
        or 4 they can be interpreted as RGB or RGBA if multichannel is
        `True`.
    multichannel : bool
        Whether the image is multichannel RGB or RGBA if multichannel. If
        not specified by user and the last dimension of the data has length
        3 or 4 it will be set as `True`. If `False` the image is
        interpreted as a luminance image.
    colormap : str, vispy.Color.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : list
        Pyramid data. List of array like image date. Each image can be N
        dimensional. If the last dimensions of the images have length 3
        or 4 they can be interpreted as RGB or RGBA if multichannel is `True`.
    metadata : dict
        Image metadata.
    multichannel : bool
        Whether the images are multichannel RGB or RGBA if multichannel. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the image is interpreted as a
        luminance image.
    colormap : 2-tuple of str, vispy.color.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance images, if the images
        are multichannel the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    contrast_limits : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image are multichannel the contrast_limits is ignored.
    contrast_limits_range : list (2,) of float
        Range for the color limits for luminace images. If the image are
        multichannel the contrast_limits_range is ignored.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported modes.

    Extended Summary
    ----------
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewer slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    _data_level : int
        Level of the currently viewed slice from the pyramid
    """

    _max_tile_shape = 1600

    def __init__(self, data, *args, **kwargs):

        self._data_level = 0
        super().__init__(np.array([np.asarray(data[-1])]), *args, **kwargs)
        self._data = data
        self._data_level = len(data) - 1
        self._top_left = np.zeros(self.ndim, dtype=int)

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    @property
    def data(self):
        """list of array: Image pyramid with base at `0`."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._update_dims()
        self.events.data()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return len(self.level_shapes[0])

    def _get_extent(self):
        return tuple((0, m) for m in self.level_shapes[0])

    def _get_range(self):
        """Shape of the base of pyramid.

        Returns
        ----------
        shape : list
            Shape of base of pyramid
        """
        return tuple((0, m, 1) for m in self.level_shapes[0])

    @property
    def data_level(self):
        """int: Current level of pyramid."""
        return self._data_level

    @data_level.setter
    def data_level(self, level):
        if self._data_level == level:
            return
        self._data_level = level
        self._set_view_slice()

    @property
    def level_shapes(self):
        """array: Shapes of each level of the pyramid."""
        if self.multichannel:
            shapes = [im.shape[:-1] for im in self.data]
        else:
            shapes = [im.shape for im in self.data]
        return np.array(shapes)

    @property
    def level_downsamples(self):
        """list: Downsample factors for each level of the pyramid."""
        return np.divide(self.level_shapes[0], self.level_shapes)

    @property
    def top_left(self):
        """tuple: Location of top left canvas pixel in image."""
        return self._top_left

    @top_left.setter
    def top_left(self, top_left):
        if np.all(self._top_left == top_left):
            return
        self._top_left = top_left.astype(int)
        self._set_view_slice()

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        nd = self.dims.not_displayed

        if self.multichannel:
            # if multichannel need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            order = self.dims.displayed_order + (self.dims.ndisplay,)
        else:
            order = self.dims.displayed_order

        # Slice thumbnail
        indices = np.array(self.dims.indices)
        downsampled = indices[nd] / self.level_downsamples[-1, nd]
        downsampled = np.round(downsampled.astype(float)).astype(int)
        downsampled = np.clip(downsampled, 0, self.level_shapes[-1, nd] - 1)
        indices[nd] = downsampled
        self._data_thumbnail = np.asarray(
            self.data[-1][tuple(indices)]
        ).transpose(order)

        # Slice currently viewed level
        indices = np.array(self.dims.indices)
        level = self.data_level
        downsampled = indices[nd] / self.level_downsamples[level, nd]
        downsampled = np.round(downsampled.astype(float)).astype(int)
        downsampled = np.clip(downsampled, 0, self.level_shapes[level, nd] - 1)
        indices[nd] = downsampled

        disp_shape = self.level_shapes[level, self.dims.displayed]
        scale = np.ones(self.ndim)
        for d in self.dims.displayed:
            scale[d] = self.level_downsamples[self.data_level][d]
        self._scale = scale
        self.events.scale()

        if np.any(disp_shape > self._max_tile_shape):
            for d in self.dims.displayed:
                indices[d] = slice(
                    self._top_left[d],
                    self._top_left[d] + self._max_tile_shape,
                    1,
                )
            self.translate = self._top_left * self.scale
        else:
            self.translate = [0] * self.ndim

        self._data_view = np.asarray(
            self.data[level][tuple(indices)]
        ).transpose(order)

        self._update_thumbnail()
        self._update_coordinates()
        self.events.set_data()

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        value : tuple
            Value of the data at the coord.
        """
        coord = np.round(self.coordinates).astype(int)
        if self.multichannel:
            shape = self._data_view.shape[:-1]
        else:
            shape = self._data_view.shape

        if all(0 <= c < s for c, s in zip(coord[self.dims.displayed], shape)):
            value = (
                self.data_level,
                self._data_view[tuple(coord[self.dims.displayed])],
            )
        else:
            value = None

        return value
