"""Image class.
"""
import types
import warnings

import numpy as np
from scipy import ndimage as ndi

from ...components.chunk import ChunkKey, ChunkRequest, chunk_loader
from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.events import Event
from ...utils.status_messages import format_float
from ..base import Layer
from ..intensity_mixin import IntensityVisualizationMixin
from ..utils.layer_utils import calc_data_range
from ._image_constants import Interpolation, Interpolation3D, Rendering
from ._image_slice import ImageSlice
from ._image_utils import guess_multiscale, guess_rgb


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

    Extended Summary
    ----------------
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

        # Initialize the current slice to an empty image.
        self._slice = ImageSlice(
            self._get_empty_image(), self._raw_to_displayed, self.rgb
        )
        self._empty = True

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

    def _get_empty_image(self):
        """Get empty image to use as the default before data is loaded.
        """
        if self.rgb:
            return np.zeros((1,) * self.dims.ndisplay + (3,))
        else:
            return np.zeros((1,) * self.dims.ndisplay)

    def _get_order(self):
        """Return the order of the displayed dimensions."""
        if self.rgb:
            # if rgb need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            return self.dims.displayed_order + (
                max(self.dims.displayed_order) + 1,
            )
        else:
            return self.dims.displayed_order

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

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = self.level_shapes[0]
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
        not_disp = self.dims.not_displayed

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
                [extent[1, ax] - 1 for ax in not_disp],
            )
        ):
            self._slice.image.raw = self._get_empty_image()
            self._slice.thumbnail.raw = self._get_empty_image()
            self._empty = True
            return
        self._empty = False

        if self.multiscale:
            # If 3d redering just show lowest level of multiscale
            if self.dims.ndisplay == 3:
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
        key = ChunkKey(self, image_indices)
        self._load_images(key, image, thumbnail_source)

    def _load_images(self, key: ChunkKey, image, thumbnail_source=None):
        """Load the image and maybe thumbnail source.

        The load will happen synchronously if any of these are true:
        1) The arrays are already ndarrays, or
        2) We are in synchronous mode, or
        3) The arrays were found in the ChunkCache.

        If the load is synchronous we immediately call our on_chunk_loaded()
        method so that we use the new data right away. If the load is
        asynchronous then sometime later our on_chunk_loaded() method will be
        called with the loaded data.
        """
        # We always load the image.
        chunks = {'image': image}

        # Optionally also load the thumbnail_source.
        if thumbnail_source is not None:
            chunks['thumbnail_source'] = thumbnail_source

        # Create the ChunkRequest for the chunks we are going to load.
        request = chunk_loader.create_request(self, key, chunks)

        # Load the chunk(s), the load could happen sync or async.
        satisfied_request = self._slice.load_chunk(request)

        if satisfied_request is None:
            # The load is going to be async, announce that a load is
            # underway, that we are no longer in the loaded state.
            self.events.loaded()
        else:
            # The load was sync, it's done, so use the chunk now.
            self.on_chunk_loaded(satisfied_request, sync=True)

    def on_chunk_loaded(
        self, request: ChunkRequest, sync: bool = False
    ) -> None:
        """The given ChunkRequest was satisfied, we can use the data now.

        This routine is called synchronously from _load_async() above, or
        it is called asynchronously sometime later when the ChunkLoader
        finishes loading the data in a worker thread or process.

        Parameters
        ----------
        request : ChunkRequest
            The request that was satisfied/loaded.
        sync : bool
            If True the chunk was loaded synchronously.
        """
        # Chunks get transposed after load.
        request.transpose_chunks(self._get_order())

        # The shape of the array after we loaded it was "wrong" in that it
        # didn't match the rgb flag that the user specified (or we inferred?).
        if self.rgb != guess_rgb(request.image.shape):
            raise ValueError(
                "Loaded chunk was the wrong shape, was rgb set correctly?"
            )

        # Pass the loaded data to the slice.
        if not self._slice.on_chunk_loaded(request):
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
            # If this specific load itself was async we need a full refresh.
            # Note: even in async mode, some loads are sync, such as cache hits
            # and loads of ndarray data.
            self.refresh()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        if not self._slice.loaded:
            # ASYNC_TODO: Do not compute the thumbnail until we are loaded.
            # Is there a nicer way to prevent this from getting called?
            return

        image = self._slice.thumbnail.view

        if self.dims.ndisplay == 3 and self.dims.ndim > 2:
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

    def _get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        -------
        value : tuple
            Value of the data at the coord.
        """
        coord = np.round(self.coordinates).astype(int)
        raw = self._slice.image.raw
        if self.rgb:
            shape = raw.shape[:-1]
        else:
            shape = raw.shape

        if all(0 <= c < s for c, s in zip(coord[self.dims.displayed], shape)):
            value = raw[tuple(coord[self.dims.displayed])]
        else:
            value = None

        if self.multiscale:
            value = (self.data_level, value)

        return value
