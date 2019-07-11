import numpy as np
from ..image import Image


class Pyramid(Image):
    """Image pyramid layer.

    Parameters
    ----------
    pyramid : list
        Pyramid data. List of array like image date. Each image can be N
        dimensional. If the last dimensions of the images have length 3
        or 4 they can be interpreted as RGB or RGBA if multichannel is
        `True`.
    metadata : dict, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel RGB or RGBA if multichannel. If
        not specified by user and the last dimension of the data has length
        3 or 4 it will be set as `True`. If `False` the image is
        interpreted as a luminance image.
    colormap : str, vispy.Color.Colormap, 2-tuple, dict, optional
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    clim : list (2,), optional
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    clim_range : list (2,), optional
        Range for the color limits. If not passed is be calculated as the
        min and max of the images. Passing a value prevents this calculation
        which can be useful when working with very large datasets that are
        dynamically loaded.
    interpolation : str, optional
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    name : str
        Name of the layer.

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
    clim : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image are multichannel the clim is ignored.
    clim_range : list (2,) of float
        Range for the color limits for luminace images. If the image are
        multichannel the clim_range is ignored.
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

    _max_tile_shape = np.array([1600, 1600])

    class_keymap = {}

    def __init__(self, pyramid, *args, **kwargs):

        with self.freeze_refresh():
            self._data_level = 0
            super().__init__(
                np.array([np.asarray(pyramid[-1])]), *args, **kwargs
            )
            self._data = pyramid
            self._data_level = len(pyramid) - 1
            self._top_left = np.array([0, 0])

            # TODO: Change dims selection when dims model changes
            self.scale = self.level_downsamples[self.data_level, [-1, -2]]

            # Re intitialize indices depending on image dims
            self._indices = (0,) * (self.ndim - 2) + (
                slice(None, None, None),
                slice(None, None, None),
            )

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

    @property
    def data(self):
        """list of array: Image pyramid with base at `0`."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

        self.refresh()

    @property
    def data_level(self):
        """int: Current level of pyramid."""
        return self._data_level

    @data_level.setter
    def data_level(self, level):
        if self._data_level == level:
            return
        self._data_level = level
        # TODO: Change dims selection when dims model changes
        self.scale = self.level_downsamples[self.data_level, [-1, -2]]
        self._top_left = self.find_top_left()
        self.refresh()

    @property
    def level_shapes(self):
        """list: Shapes of each level of the pyramid."""
        if self.multichannel:
            shapes = [im.shape[:-1] for im in self.data]
        else:
            shapes = [im.shape for im in self.data]
        return shapes

    @property
    def level_downsamples(self):
        """list: Downsample factors for each level of the pyramid."""
        return np.divide(self.level_shapes[0], self.level_shapes)

    @property
    def top_left(self):
        """2-tuple: Location of top left canvas pixel in image."""
        return self._top_left

    @top_left.setter
    def top_left(self, top_left):
        if np.all(self._top_left == top_left):
            return
        self._top_left = top_left
        self.refresh()

    def _slice_data(self):
        """Determine the slice of image from the indices."""
        indices = list(self.indices)
        top_image = self.data[-1]
        top_image_shape = self.level_shapes[-1][:-2]
        # TODO: Change dims selection when dims model changes
        rescale = self.level_downsamples[-1, :-2]
        indices[:-2] = np.round(indices[:-2] / rescale).astype(int)
        indices[:-2] = np.clip(
            indices[:-2], 0, np.subtract(top_image_shape, 1)
        )
        self._data_thumbnail = np.asarray(top_image[tuple(indices)])

        indices = list(self.indices)
        # TODO: Change dims selection when dims model changes
        rescale = self.level_downsamples[self.data_level, :-2]
        indices[:-2] = np.round(indices[:-2] / rescale).astype(int)
        indices[:-2] = np.clip(
            indices[:-2],
            0,
            np.subtract(self.level_shapes[self.data_level][:-2], 1),
        )
        if np.any(
            self.level_shapes[self.data_level][-2:] > self._max_tile_shape
        ):
            slices = [
                slice(
                    self._top_left[i],
                    self._top_left[i] + self._max_tile_shape[i],
                    1,
                )
                for i in range(2)
            ]
            indices[-2:] = slices
            self.translate = self._top_left[::-1] * self.scale[:2]
        else:
            self.translate = [0, 0]
        self._update_coordinates()

        self._data_view = np.asarray(
            self.data[self.data_level][tuple(indices)]
        )

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        self._slice_data()
        self._node.set_data(self._data_view)

        self._need_visual_update = True
        self._update()

        coord, value = self.get_value()
        self.status = self.get_message(coord, value)
        self._update_thumbnail()

    def _get_shape(self):
        """Shape of the base of pyramid.

        Returns
        ----------
        shape : list
            Shape of base of pyramid
        """
        if self.multichannel:
            return self.data[0].shape[:-1]
        return self.data[0].shape

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.
        msg : string
            String containing a message that can be used as
            a status update.
        """
        coord = np.round(self.coordinates).astype(int)
        if self.multichannel:
            shape = self._data_view.shape[:-1]
        else:
            shape = self._data_view.shape

        # TODO: Change dims selection when dims model changes
        coord[-2:] = np.clip(coord[-2:], 0, np.subtract(shape, 1))
        value = self._data_view[tuple(coord[-2:])]

        pos_in_slice = (
            self.coordinates[-2:] + self.translate[[1, 0]] / self.scale[:2]
        )

        # Make sure pos in slice doesn't go off edge
        # TODO: Change dims selection when dims model changes
        shape = self.level_shapes[self.data_level][-2:]
        pos_in_slice = np.clip(pos_in_slice, 0, np.subtract(shape, 1))
        coord[-2:] = np.round(pos_in_slice * self.scale[:2]).astype(int)

        return coord, value

    def compute_data_level(self, size):
        """Computed what level of the pyramid should be viewed given the
        current size of the requested field of view.

        Parameters
        ----------
        size : 2-tuple
            Requested size of field of view in image coordinates

        Returns
        ----------
        level : int
            Level of the pyramid to be viewing.
        """
        # Convert requested field of view from the camera into log units
        size = np.log2(np.max(size))

        # Max allowed tile in log units
        max_size = np.log2(self._max_tile_shape.max())

        # Allow for 2x coverage of field of view with max tile
        diff = size - max_size + 1

        # Find closed downsample level to diff
        # TODO: Change dims selection when dims model changes
        ds = self.level_downsamples[:, -2:].max(axis=1)
        level = np.argmin(abs(np.log2(ds) - diff))

        return level

    def find_top_left(self):
        """Finds the top left pixel of the canvas. Depends on the current
        pan and zoom position

        Returns
        ----------
        top_left : (2,) int array
            Coordinates of top left pixel.
        """

        # Find image coordinate of top left canvas pixel
        transform = self._node.canvas.scene.node_transform(self._node)
        pos = transform.map([0, 0])[:2] + self.translate[:2] / self.scale[:2]

        # TODO: Change dims selection when dims model changes
        shape = self.level_shapes[0][-2:]

        # Clip according to the max image shape
        pos = [
            np.clip(pos[1], 0, shape[0] - 1),
            np.clip(pos[0], 0, shape[1] - 1),
        ]

        # Convert to offset for image array
        top_left = np.array(pos)
        scale = self._max_tile_shape / 4
        top_left = scale * np.floor(top_left / scale)

        return top_left.astype(int)

    def get_message(self, coord, value):
        """Generate a status message based on the coordinates and information
        about what shapes are hovered over

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        value : int, float, or sequence of int or float
            Value of the data at the coord.

        Returns
        ----------
        msg : string
            String containing a message that can be used as a status update.
        """
        if isinstance(value, np.ndarray):
            if isinstance(value[0], np.integer) or isinstance(value[0], int):
                v_str = str(value)
            else:
                v_str = '[' + str.join(', ', [f'{v:0.3}' for v in value]) + ']'
        else:
            if isinstance(value, np.integer) or isinstance(value, int):
                v_str = str(value)
            else:
                v_str = f'{value:0.3}'

        msg = f'{coord}, {self.data_level}, {self.name}, value {v_str}'
        return msg

    def on_draw(self, event):
        """Called whenever the canvas is drawn, which happens whenever new
        data is sent to the canvas or the camera is moved.
        """
        size = self._parent.camera.rect.size
        data_level = self.compute_data_level(size)
        if data_level != self.data_level:
            self.data_level = data_level
        else:
            self.top_left = self.find_top_left()
