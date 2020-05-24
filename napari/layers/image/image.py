import types
import warnings
import numpy as np
from scipy import ndimage as ndi

from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.event import Event
from ...utils.status_messages import format_float
from ..base import Layer
from ..utils.layer_utils import calc_data_range
from ..intensity_mixin import IntensityVisualizationMixin
from ._image_constants import Interpolation, Interpolation3D, Rendering
from ._image_utils import guess_rgb, guess_multiscale


# Mixin must come before Layer
class Image(IntensityVisualizationMixin, Layer):
    """Image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image.
    rgb : bool
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
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
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest.

    Attributes
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a list
        and arrays are decreaing in shape then the data is treated as a
        multiscale image.
    metadata : dict
        Image metadata.
    rgb : bool
        Whether the image is rgb RGB or RGBA if rgb. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the image is interpreted as a
        luminance image.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
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

    def __init__(
        self,
        data,
        *,
        rgb=None,
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
        multiscale=None,
    ):
        if isinstance(data, types.GeneratorType):
            data = list(data)

        # Determine if data is a multiscale
        if multiscale is None:
            multiscale, data = guess_multiscale(data)

        # Determine initial shape
        if multiscale:
            init_shape = data[0].shape
        else:
            init_shape = data.shape

        # Determine if rgb
        if rgb is None:
            rgb = guess_rgb(init_shape)

        # Determine dimensionality of the data
        if rgb:
            ndim = len(init_shape) - 1
        else:
            ndim = len(init_shape)

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
            multiscale=multiscale,
        )

        self.events.add(
            interpolation=Event,
            rendering=Event,
            iso_threshold=Event,
            attenuation=Event,
        )

        # Set data
        self.rgb = rgb
        self._data = data
        if self.multiscale:
            self._data_level = len(self.data) - 1
            # Determine which level of the multiscale to use for the thumbnail.
            # Pick the smallest level with at least one axis >= 64. This is
            # done to prevent the thumbnail from being from one of the very
            # low resolution layers and therefore being very blurred.
            big_enough_levels = [
                np.any(np.greater_equal(p.shape, 64)) for p in data
            ]
            if np.any(big_enough_levels):
                self._thumbnail_level = np.where(big_enough_levels)[0][-1]
            else:
                self._thumbnail_level = 0
        else:
            self._data_level = 0
            self._thumbnail_level = 0
        self.corner_pixels[1] = self.level_shapes[self._data_level]

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
        self._interpolation = {
            2: Interpolation.NEAREST,
            3: (
                Interpolation3D.NEAREST
                if self.__class__.__name__ == 'Labels'
                else Interpolation3D.LINEAR
            ),
        }
        self.interpolation = interpolation
        self.rendering = rendering

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    def _calc_data_range(self):
        if self.multiscale:
            input_data = self.data[-1]
        else:
            input_data = self.data
        return calc_data_range(input_data)

    @property
    def dtype(self):
        return self.data[0].dtype if self.multiscale else self.data.dtype

    @property
    def data(self):
        """array: Image data."""
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

    @property
    def data_level(self):
        """int: Current level of multiscale, or 0 if image."""
        return self._data_level

    @data_level.setter
    def data_level(self, level):
        if self._data_level == level:
            return
        self._data_level = level
        self.refresh()

    @property
    def level_shapes(self):
        """array: Shapes of each level of the multiscale or just of image."""
        if self.multiscale:
            if self.rgb:
                shapes = [im.shape[:-1] for im in self.data]
            else:
                shapes = [im.shape for im in self.data]
        else:
            if self.rgb:
                shapes = [self.data.shape[:-1]]
            else:
                shapes = [self.data.shape]
        return np.array(shapes)

    @property
    def downsample_factors(self):
        """list: Downsample factors for each level of the multiscale."""
        return np.divide(self.level_shapes[0], self.level_shapes)

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
        """Return current interpolation mode.

        Selects a preset interpolation mode in vispy that determines how volume
        is displayed.  Makes use of the two Texture2D interpolation methods and
        the available interpolation methods defined in
        vispy/gloo/glsl/misc/spatial_filters.frag

        Options include:
        'bessel', 'bicubic', 'bilinear', 'blackman', 'catrom', 'gaussian',
        'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
        'nearest', 'spline16', 'spline36'

        Returns
        -------
        str
            The current interpolation mode
        """
        return str(self._interpolation[self.dims.ndisplay])

    @interpolation.setter
    def interpolation(self, interpolation):
        """Set current interpolation mode."""
        if self.dims.ndisplay == 3:
            self._interpolation[self.dims.ndisplay] = Interpolation3D(
                interpolation
            )
        else:
            self._interpolation[self.dims.ndisplay] = Interpolation(
                interpolation
            )
        self.events.interpolation()

    @property
    def rendering(self):
        """Return current rendering mode.

        Selects a preset rendering mode in vispy that determines how
        volume is displayed.  Options include:

        * ``translucent``: voxel colors are blended along the view ray until
          the result is opaque.
        * ``mip``: maxiumum intensity projection. Cast a ray and display the
          maximum value that was encountered.
        * ``additive``: voxel colors are added along the view ray until the
          result is saturated.
        * ``iso``: isosurface. Cast a ray until a certain threshold is
          encountered. At that location, lighning calculations are performed to
          give the visual appearance of a surface.
        * ``attenuated_mip``: attenuated maxiumum intensity projection. Cast a
          ray and attenuate values based on integral of encountered values,
          display the maximum value that was encountered after attenuation.
          This will make nearer objects appear more prominent.

        Returns
        -------
        str
            The current rendering mode
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        """Set current rendering mode."""
        self._rendering = Rendering(rendering)
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
                'multiscale': self.multiscale,
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

        if self.multiscale:
            # If 3d redering just show lowest level of multiscale
            if self.dims.ndisplay == 3:
                self.data_level = len(self.data) - 1

            # Slice currently viewed level
            level = self.data_level
            indices = np.array(self.dims.indices)
            downsampled_indices = (
                indices[not_disp] / self.downsample_factors[level, not_disp]
            )
            downsampled_indices = np.round(
                downsampled_indices.astype(float)
            ).astype(int)
            downsampled_indices = np.clip(
                downsampled_indices, 0, self.level_shapes[level, not_disp] - 1
            )
            indices[not_disp] = downsampled_indices

            scale = np.ones(self.ndim)
            for d in self.dims.displayed:
                scale[d] = self.downsample_factors[self.data_level][d]
            self._transforms['tile2data'].scale = scale

            if self.dims.ndisplay == 2:
                corner_pixels = np.clip(
                    self.corner_pixels,
                    0,
                    np.subtract(self.level_shapes[self.data_level], 1),
                )

                for d in self.dims.displayed:
                    indices[d] = slice(
                        corner_pixels[0, d], corner_pixels[1, d] + 1, 1
                    )
                self._transforms['tile2data'].translate = (
                    corner_pixels[0]
                    * self._transforms['data2world'].scale
                    * self._transforms['tile2data'].scale
                )

            image = np.transpose(
                np.asarray(self.data[level][tuple(indices)]), order
            )

            # Slice thumbnail
            indices = np.array(self.dims.indices)
            downsampled_indices = (
                indices[not_disp]
                / self.downsample_factors[self._thumbnail_level, not_disp]
            )
            downsampled_indices = np.round(
                downsampled_indices.astype(float)
            ).astype(int)
            downsampled_indices = np.clip(
                downsampled_indices,
                0,
                self.level_shapes[self._thumbnail_level, not_disp] - 1,
            )
            indices[not_disp] = downsampled_indices
            thumbnail_source = np.asarray(
                self.data[self._thumbnail_level][tuple(indices)]
            ).transpose(order)
        else:
            self._transforms['tile2data'].scale = np.ones(self.dims.ndim)
            image = np.asarray(self.data[self.dims.indices]).transpose(order)
            thumbnail_source = image

        if self.rgb and image.dtype.kind == 'f':
            self._data_raw = np.clip(image, 0, 1)
            self._data_view = self._raw_to_displayed(self._data_raw)
            self._data_thumbnail = self._raw_to_displayed(
                np.clip(thumbnail_source, 0, 1)
            )

        else:
            self._data_raw = image
            self._data_view = self._raw_to_displayed(self._data_raw)
            self._data_thumbnail = self._raw_to_displayed(thumbnail_source)

        if self.multiscale:
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

        if self.multiscale:
            value = (self.data_level, value)

        return value
