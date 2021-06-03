"""Image class.
"""
import types
import warnings

import numpy as np
from scipy import ndimage as ndi

from ...utils import config
from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.events import Event
from ...utils.translations import trans
from ..base import Layer
from ..intensity_mixin import IntensityVisualizationMixin
from ..utils.layer_utils import calc_data_range
from ._image_constants import Interpolation, Interpolation3D, Rendering
from ._image_slice import ImageSlice
from ._image_slice_data import ImageSliceData
from ._image_utils import guess_multiscale, guess_rgb

# Use special ChunkedSlideData for async.
if config.async_loading:
    from .experimental._chunked_slice_data import ChunkedSliceData

    SliceDataClass = ChunkedSliceData
else:
    SliceDataClass = ImageSliceData


# It is important to contain at least one abstractmethod to properly exclude this class
# in creating NAMES set inside of napari.layers.__init__
# Mixin must come before Layer
class _ImageBase(IntensityVisualizationMixin, Layer):
    """Image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image.
    rgb : bool
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
    colormap : str, napari.utils.Colormap, tuple, dict
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
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        AffineTransform object. If provided then translate, scale, rotate, and
        shear values are ignored.
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
        and arrays are decreasing in shape then the data is treated as a
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
    colormap : 2-tuple of str, napari.utils.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance images, if the image is
        rgb the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    contrast_limits : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image is rgb the contrast_limits is ignored.
    contrast_limits_range : list (2,) of float
        Range for the color limits for luminance images. If the image is
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

    Notes
    -----
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
        attenuation=0.05,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        visible=True,
        multiscale=None,
    ):
        if isinstance(data, types.GeneratorType):
            data = list(data)

        if getattr(data, 'ndim', 2) < 2:
            raise ValueError(
                trans._('Image data must have at least 2 dimensions.')
            )

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
            rotate=rotate,
            shear=shear,
            affine=affine,
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

        self._new_empty_slice()

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

    def _new_empty_slice(self):
        """Initialize the current slice to an empty image."""
        self._slice = ImageSlice(
            self._get_empty_image(), self._raw_to_displayed, self.rgb
        )
        self._empty = True

    def _get_empty_image(self):
        """Get empty image to use as the default before data is loaded."""
        if self.rgb:
            return np.zeros((1,) * self._ndisplay + (3,))
        else:
            return np.zeros((1,) * self._ndisplay)

    def _get_order(self):
        """Return the order of the displayed dimensions."""
        if self.rgb:
            # if rgb need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            return self._dims_displayed_order + (
                max(self._dims_displayed_order) + 1,
            )
        else:
            return self._dims_displayed_order

    @property
    def _data_view(self):
        """Viewable image for the current slice. (compatibility)"""
        return self._slice.image.view

    @property
    def _data_raw(self):
        """Raw image for the current slice. (compatibility)"""
        return self._slice.image.raw

    def _calc_data_range(self):
        if self.multiscale:
            input_data = self.data[-1]
        else:
            input_data = self.data
        return calc_data_range(input_data, rgb=self.rgb)

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
        self.events.data(value=self.data)
        self._set_editable()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return len(self.level_shapes[0])

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = np.subtract(self.level_shapes[0], 1)
        return np.vstack([np.zeros(len(shape)), shape])

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
        self._iso_threshold = value
        self._update_thumbnail()
        self.events.iso_threshold()

    @property
    def attenuation(self):
        """float: attenuation rate for attenuated_mip rendering."""
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value):
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
        return str(self._interpolation[self._ndisplay])

    @interpolation.setter
    def interpolation(self, interpolation):
        """Set current interpolation mode."""
        if self._ndisplay == 3:
            self._interpolation[self._ndisplay] = Interpolation3D(
                interpolation
            )
        else:
            self._interpolation[self._ndisplay] = Interpolation(interpolation)
        self.events.interpolation(value=self._interpolation[self._ndisplay])

    @property
    def rendering(self):
        """Return current rendering mode.

        Selects a preset rendering mode in vispy that determines how
        volume is displayed.  Options include:

        * ``translucent``: voxel colors are blended along the view ray until
          the result is opaque.
        * ``mip``: maximum intensity projection. Cast a ray and display the
          maximum value that was encountered.
        * ``additive``: voxel colors are added along the view ray until the
          result is saturated.
        * ``iso``: isosurface. Cast a ray until a certain threshold is
          encountered. At that location, lighning calculations are performed to
          give the visual appearance of a surface.
        * ``attenuated_mip``: attenuated maximum intensity projection. Cast a
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

    @property
    def loaded(self):
        """Has the data for this layer been loaded yet.

        With asynchronous loading the layer might exist but its data
        for the current slice has not been loaded.
        """
        return self._slice.loaded

    def _raw_to_displayed(self, raw):
        """Determine displayed image from raw image.

        For normal image layers, just return the actual image.

        Parameters
        ----------
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
        self._new_empty_slice()
        not_disp = self._dims_not_displayed

        # Check if requested slice outside of data range
        indices = np.array(self._slice_indices)
        extent = self._extent_data
        if np.any(
            np.less(
                [indices[ax] for ax in not_disp],
                [extent[0, ax] for ax in not_disp],
            )
        ) or np.any(
            np.greater(
                [indices[ax] for ax in not_disp],
                [extent[1, ax] for ax in not_disp],
            )
        ):
            return
        self._empty = False

        if self.multiscale:
            # If 3d redering just show lowest level of multiscale
            if self._ndisplay == 3:
                self.data_level = len(self.data) - 1

            # Slice currently viewed level
            level = self.data_level
            indices = np.array(self._slice_indices)
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
            for d in self._dims_displayed:
                scale[d] = self.downsample_factors[self.data_level][d]
            self._transforms['tile2data'].scale = scale

            if self._ndisplay == 2:
                for d in self._dims_displayed:
                    indices[d] = slice(
                        self.corner_pixels[0, d],
                        self.corner_pixels[1, d] + 1,
                        1,
                    )
                self._transforms['tile2data'].translate = (
                    self.corner_pixels[0] * self._transforms['tile2data'].scale
                )
            image = self.data[level][tuple(indices)]
            image_indices = indices

            # Slice thumbnail
            indices = np.array(self._slice_indices)
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

            thumbnail_source = self.data[self._thumbnail_level][tuple(indices)]
        else:
            self._transforms['tile2data'].scale = np.ones(self.ndim)
            image_indices = self._slice_indices
            image = self.data[image_indices]

            # For single-scale we don't request a separate thumbnail_source
            # from the ChunkLoader because in ImageSlice.chunk_loaded we
            # call request.thumbnail_source() and it knows to just use the
            # image itself is there is no explicit thumbnail_source.
            thumbnail_source = None

        # Load our images, might be sync or async.
        data = SliceDataClass(self, image_indices, image, thumbnail_source)
        self._load_slice(data)

    def _load_slice(self, data: SliceDataClass):
        """Load the image and maybe thumbnail source.

        Parameters
        ----------
        data : Slice
        """
        if self._slice.load(data):
            # The load was synchronous.
            self._on_data_loaded(data, sync=True)
        else:
            # The load will be asynchronous. Signal that our self.loaded
            # property is now false, since the load is in progress.
            self.events.loaded()

    def _on_data_loaded(self, data: SliceDataClass, sync: bool) -> None:
        """The given data a was loaded, use it now.

        This routine is called synchronously from _load_async() above, or
        it is called asynchronously sometime later when the ChunkLoader
        finishes loading the data in a worker thread or process.

        Parameters
        ----------
        data : ChunkRequest
            The request that was satisfied/loaded.
        sync : bool
            If True the chunk was loaded synchronously.
        """
        # Transpose after the load.
        data.transpose(self._get_order())

        # Pass the loaded data to the slice.
        if not self._slice.on_loaded(data):
            # Slice rejected it, was it for the wrong indices?
            return

        # Notify the world.
        if self.multiscale:
            self.events.scale()
            self.events.translate()

        # Announcing we are in the loaded state will make our node visible
        # if it was invisible during the load.
        self.events.loaded()

        if not sync:
            # TODO_ASYNC: Avoid calling self.refresh(), because it would
            # call our _set_view_slice(). Do we need a "refresh without
            # set_view_slice()" method that we can call?

            self.events.set_data()  # update vispy
            self._update_thumbnail()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        if not self.loaded:
            # ASYNC_TODO: Do not compute the thumbnail until we are loaded.
            # Is there a nicer way to prevent this from getting called?
            return

        image = self._slice.thumbnail.view

        if self._ndisplay == 3 and self.ndim > 2:
            image = np.max(image, axis=0)

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
            color_array = self.colormap.map(downsampled.ravel())
            colormapped = color_array.reshape(downsampled.shape + (4,))
            colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : tuple
            Value of the data.
        """
        if self.multiscale:
            # for multiscale data map the coordinate from the data back to
            # the tile
            coord = self._transforms['tile2data'].inverse(position)
        else:
            coord = position

        coord = np.round(coord).astype(int)

        raw = self._slice.image.raw
        if self.rgb:
            shape = raw.shape[:-1]
        else:
            shape = raw.shape

        if all(0 <= c < s for c, s in zip(coord[self._dims_displayed], shape)):
            value = raw[tuple(coord[self._dims_displayed])]
        else:
            value = None

        if self.multiscale:
            value = (self.data_level, value)

        return value

    # For async we add an on_chunk_loaded() method.
    if config.async_loading:
        from ...components.experimental.chunk import ChunkRequest

        def on_chunk_loaded(self, request: ChunkRequest) -> None:
            """An asynchronous ChunkRequest was loaded.

            Parameters
            ----------
            request : ChunkRequest
                This request was loaded.
            """
            # Convert the ChunkRequest to SliceData and use it.
            data = SliceDataClass.from_request(self, request)
            self._on_data_loaded(data, sync=False)


class Image(_ImageBase):
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
                'colormap': self.colormap.name,
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


if config.async_octree:
    from ..image.experimental.octree_image import _OctreeImageBase

    class Image(Image, _OctreeImageBase):
        pass
