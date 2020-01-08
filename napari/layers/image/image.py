import types
import warnings
from base64 import b64encode
from xml.etree.ElementTree import Element

import numpy as np
from imageio import imwrite
from scipy import ndimage as ndi

from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.event import Event
from ...utils.status_messages import format_float
from ..base import Layer
from ..layer_utils import calc_data_range
from ..intensity_mixin import IntensityVisualizationMixin
from ._constants import Interpolation, Rendering
from .image_utils import get_pyramid_and_rgb


# Mixin must come before Layer
class Image(IntensityVisualizationMixin, Layer):
    """Image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        an image pyramid.
    rgb : bool
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
    is_pyramid : bool
        Whether the data is an image pyramid or not. Pyramid data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be a pyramid. The first image in the list
        should be the largest.
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
    gamma : float
        Gamma correction for determining colormap linearity. Defaults to 1.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
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
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a list
        and arrays are decreaing in shape then the data is treated as an
        image pyramid.
    metadata : dict
        Image metadata.
    rgb : bool
        Whether the image is rgb RGB or RGBA if rgb. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the image is interpreted as a
        luminance image.
    is_pyramid : bool
        Whether the data is an image pyramid or not. Pyramid data is
        represented by a list of array like image data. The first image in the
        list should be the largest.
    colormap : 2-tuple of str, vispy.color.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance images, if the image is
        rgb the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    contrast_limits : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image is rgb the contrast_limits is ignored.
    contrast_limits_range : list (2,) of float
        Range for the color limits for luminace images. If the image is
        rgb the contrast_limits_range is ignored.
    gamma : float
        Gamma correction for determining colormap linearity.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.

    Extended Summary
    ----------
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewed slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    _colorbar : array
        Colorbar for current colormap.
    """

    _colormaps = AVAILABLE_COLORMAPS
    _max_tile_shape = 1600

    def __init__(
        self,
        data,
        *,
        rgb=None,
        is_pyramid=None,
        colormap='gray',
        contrast_limits=None,
        gamma=1,
        interpolation='nearest',
        rendering='mip',
        iso_threshold=0.5,
        attenuation=0.5,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
    ):
        if isinstance(data, types.GeneratorType):
            data = list(data)

        ndim, rgb, is_pyramid, data_pyramid = get_pyramid_and_rgb(
            data, pyramid=is_pyramid, rgb=rgb
        )

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
            interpolation=Event,
            rendering=Event,
            iso_threshold=Event,
            attenuation=Event,
        )

        # Set data
        self.is_pyramid = is_pyramid
        self.rgb = rgb
        self._data = data
        self._data_pyramid = data_pyramid
        self._top_left = np.zeros(ndim, dtype=int)
        if self.is_pyramid:
            self._data_level = len(data_pyramid) - 1
        else:
            self._data_level = 0

        # Intitialize image views and thumbnails with zeros
        if self.rgb:
            self._data_view = np.zeros(
                (1,) * self.dims.ndisplay + (self.shape[-1],)
            )
        else:
            self._data_view = np.zeros((1,) * self.dims.ndisplay)
        self._data_raw = self._data_view
        self._data_thumbnail = self._data_view

        # Set contrast_limits and colormaps
        self._gamma = gamma
        self._iso_threshold = iso_threshold
        self._attenuation = attenuation
        if contrast_limits is None:
            self.contrast_limits_range = self._calc_data_range()
        else:
            self.contrast_limits_range = contrast_limits
        self._contrast_limits = tuple(self.contrast_limits_range)
        self.colormap = colormap
        self.contrast_limits = self._contrast_limits
        self.interpolation = interpolation
        self.rendering = rendering

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    def _calc_data_range(self):
        if self.is_pyramid:
            input_data = self._data_pyramid[-1]
        else:
            input_data = self.data
        return calc_data_range(input_data)

    @property
    def dtype(self):
        return self.data[0].dtype if self.is_pyramid else self.data.dtype

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        ndim, rgb, is_pyramid, data_pyramid = get_pyramid_and_rgb(
            data, pyramid=self.is_pyramid, rgb=self.rgb
        )
        self.is_pyramid = is_pyramid
        self.rgb = rgb
        self._data = data
        self._data_pyramid = data_pyramid

        self._update_dims()
        self.events.data()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return len(self.level_shapes[0])

    def _get_extent(self):
        return tuple((0, m) for m in self.level_shapes[0])

    @property
    def data_level(self):
        """int: Current level of pyramid, or 0 if image."""
        return self._data_level

    @data_level.setter
    def data_level(self, level):
        if self._data_level == level:
            return
        self._data_level = level
        self.refresh()

    @property
    def level_shapes(self):
        """array: Shapes of each level of the pyramid or just of image."""
        if self.is_pyramid:
            if self.rgb:
                shapes = [im.shape[:-1] for im in self._data_pyramid]
            else:
                shapes = [im.shape for im in self._data_pyramid]
        else:
            if self.rgb:
                shapes = [self.data.shape[:-1]]
            else:
                shapes = [self.data.shape]
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
        self.refresh()

    @property
    def iso_threshold(self):
        """float: threshold for isosurface."""
        return self._iso_threshold

    @iso_threshold.setter
    def iso_threshold(self, value):
        self.status = format_float(value)
        self._iso_threshold = value
        self._update_thumbnail()
        self.events.iso_threshold()

    @property
    def attenuation(self):
        """float: attenuation rate for attenuated_mip rendering."""
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value):
        self.status = format_float(value)
        self._attenuation = value
        self._update_thumbnail()
        self.events.attenuation()

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
            * attenuated_mip: attenuated maxiumum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        if isinstance(rendering, str):
            rendering = Rendering(rendering)

        self._rendering = rendering
        self.events.rendering()

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'rgb': self.rgb,
                'is_pyramid': self.is_pyramid,
                'colormap': self.colormap[0],
                'contrast_limits': self.contrast_limits,
                'interpolation': self.interpolation,
                'rendering': self.rendering,
                'iso_threshold': self.iso_threshold,
                'attenuation': self.attenuation,
                'gamma': self.gamma,
                'data': self.data,
            }
        )
        return state

    def _raw_to_displayed(self, raw):
        """Determine displayed image from raw image.

        For normal image layers, just return the actual image.

        Parameters
        -------
        raw : array
            Raw array.

        Returns
        -------
        image : array
            Displayed array.
        """
        image = raw
        return image

    def _set_view_slice(self):
        """Set the view given the indices to slice with."""
        not_disp = self.dims.not_displayed

        if self.rgb:
            # if rgb need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            order = self.dims.displayed_order + (
                max(self.dims.displayed_order) + 1,
            )
        else:
            order = self.dims.displayed_order

        if self.is_pyramid:
            # If 3d redering just show lowest level of pyramid
            if self.dims.ndisplay == 3:
                self.data_level = len(self._data_pyramid) - 1

            # Slice currently viewed level
            level = self.data_level
            indices = np.array(self.dims.indices)
            downsampled_indices = (
                indices[not_disp] / self.level_downsamples[level, not_disp]
            )
            downsampled_indices = np.round(
                downsampled_indices.astype(float)
            ).astype(int)
            downsampled_indices = np.clip(
                downsampled_indices, 0, self.level_shapes[level, not_disp] - 1
            )
            indices[not_disp] = downsampled_indices

            disp_shape = self.level_shapes[level, self.dims.displayed]
            scale = np.ones(self.ndim)
            for d in self.dims.displayed:
                scale[d] = self.level_downsamples[self.data_level][d]
            self._scale_view = scale

            if np.any(disp_shape > self._max_tile_shape):
                for d in self.dims.displayed:
                    indices[d] = slice(
                        self._top_left[d],
                        self._top_left[d] + self._max_tile_shape,
                        1,
                    )
                self._translate_view = (
                    self._top_left * self.scale * self._scale_view
                )
            else:
                self._translate_view = [0] * self.ndim

            image = np.asarray(
                self._data_pyramid[level][tuple(indices)]
            ).transpose(order)

            if level == len(self._data_pyramid) - 1:
                thumbnail = image
            else:
                # Slice thumbnail
                indices = np.array(self.dims.indices)
                downsampled_indices = (
                    indices[not_disp] / self.level_downsamples[-1, not_disp]
                )
                downsampled_indices = np.round(
                    downsampled_indices.astype(float)
                ).astype(int)
                downsampled_indices = np.clip(
                    downsampled_indices, 0, self.level_shapes[-1, not_disp] - 1
                )
                indices[not_disp] = downsampled_indices
                thumbnail = np.asarray(
                    self._data_pyramid[-1][tuple(indices)]
                ).transpose(order)
        else:
            self._scale_view = np.ones(self.dims.ndim)
            image = np.asarray(self.data[self.dims.indices]).transpose(order)
            thumbnail = image

        if self.rgb and image.dtype.kind == 'f':
            self._data_raw = np.clip(image, 0, 1)
            self._data_view = self._raw_to_displayed(self._data_raw)
            self._data_thumbnail = self._raw_to_displayed(
                np.clip(thumbnail, 0, 1)
            )

        else:
            self._data_raw = image
            self._data_view = self._raw_to_displayed(self._data_raw)
            self._data_thumbnail = self._raw_to_displayed(thumbnail)

        if self.is_pyramid:
            self.events.scale()
            self.events.translate()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        if self.dims.ndisplay == 3 and self.dims.ndim > 2:
            image = np.max(self._data_thumbnail, axis=0)
        else:
            image = self._data_thumbnail

        # float16 not supported by ndi.zoom
        dtype = np.dtype(image.dtype)
        if dtype in [np.dtype(np.float16)]:
            image = image.astype(np.float32)

        raw_zoom_factor = np.divide(
            self._thumbnail_shape[:2], image.shape[:2]
        ).min()
        new_shape = np.clip(
            raw_zoom_factor * np.array(image.shape[:2]),
            1,  # smallest side should be 1 pixel wide
            self._thumbnail_shape[:2],
        )
        zoom_factor = tuple(new_shape / image.shape[:2])
        if self.rgb:
            # warning filter can be removed with scipy 1.4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downsampled = ndi.zoom(
                    image, zoom_factor + (1,), prefilter=False, order=0
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
            downsampled = downsampled ** self.gamma
            color_array = self.colormap[1][downsampled.ravel()]
            colormapped = color_array.rgba.reshape(downsampled.shape + (4,))
            colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def _get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        value : tuple
            Value of the data at the coord.
        """
        coord = np.round(self.coordinates).astype(int)
        if self.rgb:
            shape = self._data_raw.shape[:-1]
        else:
            shape = self._data_raw.shape

        if all(0 <= c < s for c, s in zip(coord[self.dims.displayed], shape)):
            value = self._data_raw[tuple(coord[self.dims.displayed])]
        else:
            value = None

        if self.is_pyramid:
            value = (self.data_level, value)

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
        mapped_image = self.colormap[1][image.ravel()]
        mapped_image = mapped_image.RGBA.reshape(image.shape + (4,))
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
