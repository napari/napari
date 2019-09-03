import warnings
from xml.etree.ElementTree import Element
from base64 import b64encode
from imageio import imwrite
import numpy as np
from copy import copy
from scipy import ndimage as ndi
import vispy.color
from ..base import Layer
from ...util.misc import (
    is_multichannel,
    calc_data_range,
    increment_unnamed_colormap,
)
from ...util.event import Event
from ...util.status_messages import format_float
from ._constants import Rendering, Interpolation, AVAILABLE_COLORMAPS


class Image(Layer):
    """Image layer.

    Parameters
    ----------
    data : array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if multichannel is `True`.
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
    contrast_limits : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image is multichannel the contrast_limits is ignored.
    contrast_limits_range : list (2,) of float
        Range for the color limits for luminace images. If the image is
        multichannel the contrast_limits_range is ignored.
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

    def __init__(
        self,
        data,
        *,
        multichannel=None,
        colormap='gray',
        contrast_limits=None,
        interpolation='nearest',
        rendering='mip',
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
    ):
        # Determine if multichannel, and determine dimensionality
        if multichannel is False:
            self._multichannel = multichannel
        else:
            # If multichannel is True or None then guess if multichannel
            # allowed or not, and if allowed set it to be True
            self._multichannel = is_multichannel(data.shape)

        if self.multichannel:
            ndim = data.ndim - 1
        else:
            ndim = data.ndim

        super().__init__(
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self.events.add(
            contrast_limits=Event,
            colormap=Event,
            interpolation=Event,
            rendering=Event,
        )

        # Set data
        self._data = data

        # Intitialize image views and thumbnails with zeros
        if self.multichannel:
            self._data_view = np.zeros(
                (1,) * self.dims.ndisplay + (self.shape[-1],)
            )
        else:
            self._data_view = np.zeros((1,) * self.dims.ndisplay)
        self._data_thumbnail = self._data_view

        # Set contrast_limits and colormaps
        self._colormap_name = ''
        self._contrast_limits_msg = ''
        if contrast_limits is None:
            self._contrast_limits_range = calc_data_range(self.data)
        else:
            self._contrast_limits_range = contrast_limits
        self._contrast_limits = copy(self._contrast_limits_range)
        self.colormap = colormap
        self.contrast_limits = self._contrast_limits
        self.interpolation = interpolation
        self.rendering = rendering

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if self.multichannel:
            self._multichannel = is_multichannel(data.shape)
        self._update_dims()
        self.events.data()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        if self.multichannel:
            ndim = self.data.ndim - 1
        else:
            ndim = self.data.ndim
        return ndim

    def _get_extent(self):
        if self.multichannel:
            shape = self.data.shape[:-1]
        else:
            shape = self.data.shape

        return tuple((0, m) for m in shape)

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
        self._set_view_slice()

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
        self._cmap = self._colormaps[name]
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    @property
    def contrast_limits(self):
        """list of float: Limits to use for the colormap."""
        return list(self._contrast_limits)

    @contrast_limits.setter
    def contrast_limits(self, contrast_limits):
        self._contrast_limits_msg = (
            format_float(contrast_limits[0])
            + ', '
            + format_float(contrast_limits[1])
        )
        self.status = self._contrast_limits_msg
        self._contrast_limits = contrast_limits
        if contrast_limits[0] < self._contrast_limits_range[0]:
            self._contrast_limits_range[0] = copy(contrast_limits[0])
        if contrast_limits[1] > self._contrast_limits_range[1]:
            self._contrast_limits_range[1] = copy(contrast_limits[1])
        self._update_thumbnail()
        self.events.contrast_limits()

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
        self.events.interpolation()

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
        if self.multichannel:
            # if multichannel need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            order = self.dims.displayed_order + (self.dims.ndisplay,)
        else:
            order = self.dims.displayed_order

        self._data_view = np.asarray(self.data[self.dims.indices]).transpose(
            order
        )

        self._data_thumbnail = self._data_view
        self._update_thumbnail()
        self._update_coordinates()
        self.events.set_data()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        if self.dims.ndisplay == 3:
            image = np.max(self._data_thumbnail, axis=0)
        else:
            image = self._data_thumbnail

        # float16 not supported by ndi.zoom
        dtype = np.dtype(image.dtype)
        if dtype in [np.dtype(np.float16)]:
            image = image.astype(np.float32)

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
            low, high = self.contrast_limits
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
        if self.multichannel:
            shape = self._data_view.shape[:-1]
        else:
            shape = self._data_view.shape

        if all(0 <= c < s for c, s in zip(coord[self.dims.displayed], shape)):
            value = self._data_view[tuple(coord[self.dims.displayed])]
        else:
            value = None

        return value

    def to_xml_list(self):
        """Generates a list with a single xml element that defines the
        currently viewed image as a png according to the svg specification.

        Returns
        ----------
        xml : list of xml.etree.ElementTree.Element
            List of a single xml element specifying the currently viewed image
            as a png according to the svg specification.
        """
        if self.dims.ndisplay == 3:
            image = np.max(self._data_thumbnail, axis=0)
        else:
            image = self._data_thumbnail
        image = np.clip(
            image, self.contrast_limits[0], self.contrast_limits[1]
        )
        image = image - self.contrast_limits[0]
        color_range = self.contrast_limits[1] - self.contrast_limits[0]
        if color_range != 0:
            image = image / color_range
        mapped_image = (self.colormap[1].map(image) * 255).astype('uint8')
        mapped_image = mapped_image.reshape(list(self._data_view.shape) + [4])
        image_str = imwrite('<bytes>', mapped_image, format='png')
        image_str = "data:image/png;base64," + str(b64encode(image_str))[2:-1]
        props = {'xlink:href': image_str}
        width = str(self.shape[self.dims.displayed[1]])
        height = str(self.shape[self.dims.displayed[0]])
        opacity = str(self.opacity)
        xml = Element(
            'image', width=width, height=height, opacity=opacity, **props
        )
        return [xml]
