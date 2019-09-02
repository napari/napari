from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
from copy import copy
import vispy.color
from ..base import Layer
from ...util.event import Event
from ..image._constants import Rendering, Interpolation, AVAILABLE_COLORMAPS
from ...util.status_messages import format_float
from ...util.misc import calc_data_range, increment_unnamed_colormap
from vispy.color import get_color_names, Color


class Surface(Layer):
    """
    Surface layer renders meshes onto the canvas.

    Parameters
    ----------
    data : 3-tuple
        A pair of an (N, D) array of vertices and an (M, 3) array of
        indices of triangles, and a (N,) array of values to color vertices
        with.
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
    data : 3-tuple
        A pair of an (N, D) array of vertices and an (M, 3) array of
        indices of triangles, and a (N,) array of values to color vertices
        with.
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

    Extended Summary
    ----------
    _data_view : (M, 2, 2) array
        The start point and projections of N vectors in 2D for vectors whose
        start point is in the currently viewed slice.
    _mesh_vertices : (4N, 2) array
        The four corner points for the mesh representation of each vector as as
        rectangle in the slice that it starts in.
    _mesh_triangles : (2N, 3) array
        The integer indices of the `_mesh_vertices` that form the two triangles
        for the mesh representation of the vectors.
    """

    _colormaps = AVAILABLE_COLORMAPS

    def __init__(
        self,
        data,
        *,
        colormap='gray',
        contrast_limits=None,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
    ):

        ndim = data[0].shape[1]

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

        # Save the vector style params
        # Set contrast_limits and colormaps
        self._colormap_name = ''
        self._contrast_limits_msg = ''
        if contrast_limits is None:
            self._contrast_limits_range = calc_data_range(data[2])
        else:
            self._contrast_limits_range = contrast_limits
        self._contrast_limits = copy(self._contrast_limits_range)
        self.colormap = colormap
        self.contrast_limits = self._contrast_limits
        self.interpolation = 'nearest'
        self.rendering = 'mip'

        # Data containing vectors in the currently viewed slice
        self._data_view = np.empty((0, 2))

        # assign mesh data and establish default behavior
        self.data = data

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        """3-tuple of vertices, triangles, and values."""

        self._data = data

        self._update_dims()
        self.events.data()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return self.data[0].shape[1]

    def _get_extent(self):
        """Determine ranges for slicing given by (min, max, step)."""
        if len(self.data) == 0:
            maxs = np.ones(self.data[0].shape[1], dtype=int)
            mins = np.zeros(self.data[0].shape[1], dtype=int)
        else:
            maxs = np.max(self.data[0], axis=0)
            mins = np.min(self.data[0], axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

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
        """Sets the view given the indices to slice with."""

        vertices = self.data[0]
        not_disp = list(self.dims.not_displayed)
        disp = list(self.dims.displayed)
        indices = np.array(self.dims.indices)

        # if len(vertices) == 0:
        #     self._view_faces = None
        #     self._data_view = np.zeros((0, self.dims.ndisplay))
        # elif self.ndim > self.dims.ndisplay:
        #     data = self.data[0][:, not_disp].astype('int')
        #     matches = np.all(data == indices[not_disp], axis=1)
        #     matches = np.where(matches)[0]
        #     self._data_view = self.data[0][np.ix_(matches, disp)]
        #     if len(matches) == 0:
        #         self._view_faces = None
        #     else:
        #         self._view_faces = self.data[1]
        # else:
        self._view_faces = self.data[1]
        self._view_values = self.data[2]
        self._data_view = self.data[0][:, disp]

        self._update_thumbnail()
        self._update_coordinates()
        self.events.set_data()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        # calculate min vals for the vertices and pad with 0.5
        # the offset is needed to ensure that the top left corner of the
        # vectors corresponds to the top left corner of the thumbnail
        # offset = (
        #     np.array([self.dims.range[d][0] for d in self.dims.displayed])
        #     + 0.5
        # )[-2:]
        # # calculate range of values for the vertices and pad with 1
        # # padding ensures the entire vector can be represented in the thumbnail
        # # without getting clipped
        # shape = np.ceil(
        #     [
        #         self.dims.range[d][1] - self.dims.range[d][0] + 1
        #         for d in self.dims.displayed
        #     ]
        # ).astype(int)[-2:]
        # zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()
        #
        # vectors = copy(self._data_view[:, :, -2:])
        # vectors[:, 1, :] = vectors[:, 0, :] + vectors[:, 1, :] * self.length
        # downsampled = (vectors - offset) * zoom_factor
        # downsampled = np.clip(
        #     downsampled, 0, np.subtract(self._thumbnail_shape[:2], 1)
        # )
        # colormapped = np.zeros(self._thumbnail_shape)
        # colormapped[..., 3] = 1
        # col = Color(self.edge_color).rgba
        # if len(downsampled) > self._max_vectors_thumbnail:
        #     inds = np.random.randint(
        #         0, len(downsampled), self._max_vectors_thumbnail
        #     )
        #     downsampled = downsampled[inds]
        # for v in downsampled:
        #     start = v[0]
        #     stop = v[1]
        #     step = int(np.ceil(np.max(abs(stop - start))))
        #     x_vals = np.linspace(start[0], stop[0], step)
        #     y_vals = np.linspace(start[1], stop[1], step)
        #     for x, y in zip(x_vals, y_vals):
        #         colormapped[int(x), int(y), :] = col
        # colormapped[..., 3] *= self.opacity
        # self.thumbnail = colormapped
        pass

    def to_xml_list(self):
        """Convert vectors to a list of svg xml elements.

        Returns
        ----------
        xml : list
            List of xml elements defining each vector as a line according to
            the svg specification.
        """
        xml_list = []

        width = str(self.edge_width)
        edge_color = (255 * Color(self.edge_color).rgba).astype(np.int)
        stroke = f'rgb{tuple(edge_color[:3])}'
        opacity = str(self.opacity)
        props = {'stroke': stroke, 'stroke-width': width, 'opacity': opacity}

        for v in self._data_view:
            x1 = str(v[0, -2])
            y1 = str(v[0, -1])
            x2 = str(v[0, -2] + self.length * v[1, -2])
            y2 = str(v[0, -1] + self.length * v[1, -1])

            element = Element('line', x1=y1, y1=x1, x2=y2, y2=x2, **props)
            xml_list.append(element)

        return xml_list

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        value : int, None
            Value of the data at the coord.
        """

        return None
