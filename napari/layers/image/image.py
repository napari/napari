import warnings
from xml.etree.ElementTree import Element
from base64 import b64encode
from imageio import imwrite
import numpy as np
from copy import copy
from scipy import ndimage as ndi
import vispy.color
from ..base import Layer
from vispy.scene.visuals import Image as ImageNode
from ...util.misc import (
    is_multichannel,
    calc_data_range,
    increment_unnamed_colormap,
)
from ...util.event import Event
from ._constants import Interpolation, AVAILABLE_COLORMAPS


class Image(Layer):
    """Image layer.

    Parameters
    ----------
    image : array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if multichannel is `True`.
    metadata : dict
        Image metadata.
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
    clim : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    clim_range : list (2,)
        Range for the color limits. If not passed is be calculated as the
        min and max of the image. Passing a value prevents this calculation
        which can be useful when working with very large datasets that are
        dynamically loaded.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    name : str
        Name of the layer.

    Attributes
    ----------
    data : array
        Image data. Can be N dimensional. If the last dimension has length 3
        or 4 can be interpreted as RGB or RGBA if multichannel is `True`.
    metadata : dict
        Image metadata.
    multichannel : bool
        Whether the image is multichannel RGB or RGBA if multichannel. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the image is interpreted as a
        luminance image.
    colormap : 2-tuple of str, vispy.color.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance images, if the image is
        multichannel the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    clim : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image is multichannel the clim is ignored.
    clim_range : list (2,) of float
        Range for the color limits for luminace images. If the image is
        multichannel the clim_range is ignored.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported modes.

    Extended Summary
    ----------
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewed slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    """

    _colormaps = AVAILABLE_COLORMAPS

    class_keymap = {}

    default_interpolation = str(Interpolation.NEAREST)

    def __init__(
        self,
        image,
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

        visual = ImageNode(None, method='auto')
        super().__init__(visual, name)

        self.events.add(clim=Event, colormap=Event, interpolation=Event)

        with self.freeze_refresh():
            # Set data
            self._data = image
            self.metadata = metadata or {}
            self.multichannel = multichannel

            # Intitialize image views and thumbnails with zeros
            if self.multichannel:
                self._data_view = np.zeros((1, 1) + (self.shape[-1],))
            else:
                self._data_view = np.zeros((1, 1))
            self._data_thumbnail = self._data_view

            # Set clims and colormaps
            self._colormap_name = ''
            self._clim_msg = ''
            if clim_range is None:
                self._clim_range = calc_data_range(self.data)
            else:
                self._clim_range = clim_range
            if clim is None:
                self.clim = copy(self._clim_range)
            else:
                self.clim = clim
            self.colormap = colormap
            self.interpolation = interpolation

            # Set update flags
            self._need_display_update = False
            self._need_visual_update = False

            # Re intitialize indices depending on image dims
            self._indices = (0,) * (self.ndim - 2) + (
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
        self._node._cmap = self._colormaps[name]
        self.refresh()
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

    @property
    def interpolation(self):
        """{
            'bessel', 'bicubic', 'bilinear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
            }: Equipped interpolation method's name.
        """
        return str(self._interpolation)

    @interpolation.setter
    def interpolation(self, interpolation):
        if isinstance(interpolation, str):
            interpolation = Interpolation(interpolation)
        self._interpolation = interpolation
        self._node.interpolation = interpolation.value
        self.events.interpolation()

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        indices = list(self.indices)
        indices[:-2] = np.clip(
            indices[:-2], 0, np.subtract(self.shape[:-2], 1)
        )
        self._data_view = np.asarray(self.data[tuple(indices)])

        self._node.set_data(self._data_view)

        self._need_visual_update = True
        self._update()

        coord, value = self.get_value()
        self.status = self.get_message(coord, value)

        self._data_thumbnail = self._data_view
        self._update_thumbnail()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        image = self._data_thumbnail
        zoom_factor = np.divide(
            self._thumbnail_shape[:2], image.shape[:2]
        ).min()
        if self.multichannel:
            # warning filter can be removed with scipy 1.4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downsampled = ndi.zoom(
                    image,
                    (zoom_factor, zoom_factor, 1),
                    prefilter=False,
                    order=0,
                )
            if image.shape[2] == 4:  # image is RGBA
                colormapped = np.copy(downsampled)
                colormapped[..., 3] = downsampled[..., 3] * self.opacity
                if downsampled.dtype == np.uint8:
                    colormapped = colormapped.astype(np.uint8)
            else:  # image is RGB
                if downsampled.dtype == np.uint8:
                    alpha = np.full(
                        downsampled.shape[:2] + (1,),
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
        coord : 2-tuple of int
            Position of cursor in image space.
        value : int, float, or sequence of int or float
            Value of the data at the coord.
        """
        coord = np.round(self.coordinates).astype(int)
        if self.multichannel:
            shape = self._data_view.shape[:-1]
        else:
            shape = self._data_view.shape
        coord[-2:] = np.clip(coord[-2:], 0, np.asarray(shape) - 1)

        value = self._data_view[tuple(coord[-2:])]

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
            if isinstance(value, (np.integer, np.bool_)):
                msg = msg + str(value)
            else:
                msg = msg + f'{value:0.3}'

        return msg

    def to_xml_list(self):
        """Generates a list with a single xml element that defines the
        currently viewed image as a png according to the svg specification.

        Returns
        ----------
        xml : list of xml.etree.ElementTree.Element
            List of a single xml element specifying the currently viewed image
            as a png according to the svg specification.
        """
        image = np.clip(self._data_view, self.clim[0], self.clim[1])
        image = image - self.clim[0]
        color_range = self.clim[1] - self.clim[0]
        if color_range != 0:
            image = image / color_range
        mapped_image = (self.colormap[1].map(image) * 255).astype('uint8')
        mapped_image = mapped_image.reshape(list(self._data_view.shape) + [4])
        image_str = imwrite('<bytes>', mapped_image, format='png')
        image_str = "data:image/png;base64," + str(b64encode(image_str))[2:-1]
        props = {'xlink:href': image_str}
        width = str(self.shape[-1])
        height = str(self.shape[-2])
        opacity = str(self.opacity)
        xml = Element(
            'image', width=width, height=height, opacity=opacity, **props
        )
        return [xml]

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord, value = self.get_value()
        self.status = self.get_message(coord, value)
