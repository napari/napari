import warnings
from collections import deque
from contextlib import contextmanager
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from napari.layers.base import Layer, no_op
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.image._image_utils import guess_multiscale
from napari.layers.image.image import _ImageBase
from napari.layers.labels._labels_constants import (
    LabelColorMode,
    LabelsRendering,
    Mode,
)
from napari.layers.labels._labels_mouse_bindings import draw, pick
from napari.layers.labels._labels_utils import (
    expand_slice,
    get_contours,
    indices_in_shape,
    interpolate_coordinates,
    sphere_indices,
)
from napari.layers.utils.color_transformations import transform_color
from napari.layers.utils.layer_utils import _FeatureTable
from napari.utils import config
from napari.utils._dtype import normalize_dtype, vispy_texture_dtype
from napari.utils.colormaps import (
    direct_colormap,
    ensure_colormap,
    label_colormap,
)
from napari.utils.events import EmitterGroup, Event
from napari.utils.events.custom_types import Array
from napari.utils.geometry import clamp_point_to_bounding_box
from napari.utils.indexing import index_in_slice
from napari.utils.migrations import deprecated_constructor_arg_by_attr
from napari.utils.misc import _is_array_type
from napari.utils.naming import magic_name
from napari.utils.status_messages import generate_layer_coords_status
from napari.utils.translations import trans


class Labels(_ImageBase):
    """Labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or multiscale. Must be integer type or bools.
        Please note multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    num_colors : int
        Number of unique colors to use in colormap.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    properties : dict {str: array (N,)} or DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays.
    seed : float
        Seed for colormap random generator.
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
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    rendering : str
        3D Rendering mode used by vispy. Must be one {'translucent', 'iso_categorical'}.
        'translucent' renders without lighting. 'iso_categorical' uses isosurface
        rendering to calculate lighting effects on labeled surfaces.
        The default value is 'iso_categorical'.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    visible : bool
        Whether the layer visual is currently being displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.

    Attributes
    ----------
    data : array or list of array
        Integer label data as an array or multiscale. Can be N dimensional.
        Every pixel contains an integer ID corresponding to the region it
        belongs to. The label 0 is rendered as transparent. Please note
        multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap.
    features : Dataframe-like
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays. While there is no limit to the number of custom labels, the
        the layer will render incorrectly if they map to more than 1024 distinct
        colors.
    seed : float
        Seed for colormap random generator.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    contiguous : bool
        If `True`, the fill bucket changes only connected pixels of same label.
    n_edit_dimensions : int
        The number of dimensions across which labels will be edited.
    contour : int
        If greater than 0, displays contours of labels instead of shaded regions
        with a thickness equal to its value. Must be >= 0.
    brush_size : float
        Size of the paint brush in data coordinates.
    selected_label : int
        Index of selected label. Can be greater than the current maximum label.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICK mode the cursor functions like a color picker, setting the
        clicked on label to be the current label. If the background is picked it
        will select the background label `0`.

        In PAINT mode the cursor functions like a paint brush changing any
        pixels it brushes over to the current label. If the background label
        `0` is selected than any pixels will be changed to background and this
        tool functions like an eraser. The size and shape of the cursor can be
        adjusted in the properties widget.

        In FILL mode the cursor functions like a fill bucket replacing pixels
        of the label clicked on with the current label. It can either replace
        all pixels of that label or just those that are contiguous with the
        clicked on pixel. If the background label `0` is selected than any
        pixels will be changed to background and this tool functions like an
        eraser.

        In ERASE mode the cursor functions similarly to PAINT mode, but to
        paint with background label, which effectively removes the label.
    plane : SlicingPlane
        Properties defining plane rendering in 3D.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.

    Notes
    -----
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    """

    events: EmitterGroup

    _modeclass = Mode

    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
        Mode.PICK: pick,
        Mode.PAINT: draw,
        Mode.FILL: draw,
        Mode.ERASE: draw,
    }

    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
        Mode.PICK: no_op,
        Mode.PAINT: no_op,
        Mode.FILL: no_op,
        Mode.ERASE: no_op,
    }

    _cursor_modes = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.PICK: 'cross',
        Mode.PAINT: 'circle',
        Mode.FILL: 'cross',
        Mode.ERASE: 'circle',
    }

    _history_limit = 100

    @deprecated_constructor_arg_by_attr("seed")
    def __init__(
        self,
        data,
        *,
        num_colors=49,
        features=None,
        properties=None,
        color=None,
        seed_rng=None,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending='translucent',
        rendering='iso_categorical',
        depiction='volume',
        visible=True,
        multiscale=None,
        cache=True,
        plane=None,
        experimental_clipping_planes=None,
    ) -> None:
        if name is None and data is not None:
            name = magic_name(data)

        self._seed = 0.5
        self._seed_rng: Optional[int] = seed_rng
        self._background_label = 0
        self._num_colors = num_colors
        self._random_colormap = label_colormap(
            self.num_colors, self.seed, self._background_label
        )
        self._direct_colormap = direct_colormap()
        self._color_mode = LabelColorMode.AUTO
        self._show_selected_label = False
        self._contour = 0
        self._cached_labels = None
        self._cached_mapped_labels = None

        data = self._ensure_int_labels(data)

        super().__init__(
            data,
            rgb=False,
            colormap=self._random_colormap,
            contrast_limits=[0.0, 2**23 - 1.0],
            interpolation2d='nearest',
            interpolation3d='nearest',
            rendering=rendering,
            depiction=depiction,
            iso_threshold=0,
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
            cache=cache,
            plane=plane,
            experimental_clipping_planes=experimental_clipping_planes,
        )

        self.events.add(
            preserve_labels=Event,
            show_selected_label=Event,
            properties=Event,
            n_edit_dimensions=Event,
            contiguous=Event,
            brush_size=Event,
            selected_label=Event,
            color_mode=Event,
            brush_shape=Event,
            contour=Event,
            features=Event,
            paint=Event,
            labels_update=Event,
        )

        self._feature_table = _FeatureTable.from_layer(
            features=features, properties=properties
        )
        self._label_index = self._make_label_index()

        self._n_edit_dimensions = 2
        self._contiguous = True
        self._brush_size = 10

        self._selected_label = 1
        self.colormap.selection = self._selected_label
        self.colormap.use_selection = self._show_selected_label
        self._selected_color = self.get_color(self._selected_label)
        self._updated_slice = None
        self.color = color

        self._status = self.mode
        self._preserve_labels = False

        self._reset_history()

        # Trigger generation of view slice and thumbnail
        self.refresh()
        self._reset_editable()

    @property
    def rendering(self):
        """Return current rendering mode.

        Selects a preset rendering mode in vispy that determines how
        lablels are displayed.  Options include:

        * ``translucent``: voxel colors are blended along the view ray until
          the result is opaque.
        * ``iso_categorical``: isosurface for categorical data.
          Cast a ray until a non-background value is encountered. At that
          location, lighning calculations are performed to give the visual
          appearance of a surface.

        Returns
        -------
        str
            The current rendering mode
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        self._rendering = LabelsRendering(rendering)
        self.events.rendering()

    @property
    def contiguous(self):
        """bool: fill bucket changes only connected pixels of same label."""
        return self._contiguous

    @contiguous.setter
    def contiguous(self, contiguous):
        self._contiguous = contiguous
        self.events.contiguous()

    @property
    def n_edit_dimensions(self):
        return self._n_edit_dimensions

    @n_edit_dimensions.setter
    def n_edit_dimensions(self, n_edit_dimensions):
        self._n_edit_dimensions = n_edit_dimensions
        self.events.n_edit_dimensions()

    @property
    def contour(self) -> int:
        """int: displays contours of labels instead of shaded regions."""
        return self._contour

    @contour.setter
    def contour(self, contour: int) -> None:
        if contour < 0:
            raise ValueError("contour value must be >= 0")
        self._contour = int(contour)
        self.events.contour()
        self.refresh()

    @property
    def brush_size(self):
        """float: Size of the paint in world coordinates."""
        return self._brush_size

    @brush_size.setter
    def brush_size(self, brush_size):
        self._brush_size = int(brush_size)
        self.cursor_size = self._calculate_cursor_size()
        self.events.brush_size()

    def _calculate_cursor_size(self):
        # Convert from brush size in data coordinates to
        # cursor size in world coordinates
        scale = self._data_to_world.scale
        min_scale = np.min(
            [abs(scale[d]) for d in self._slice_input.displayed]
        )
        return abs(self.brush_size * min_scale)

    @property
    def seed(self):
        """float: Seed for colormap random generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        warnings.warn(
            "seed is deprecated since 0.4.19 and will be removed in 0.5.0, please use seed_rng instead",
            FutureWarning,
            stacklevel=2,
        )

        self._seed = seed
        self.colormap = label_colormap(
            self.num_colors, self.seed, self._background_label
        )
        self._cached_labels = None  # invalidate the cached color mapping
        self._selected_color = self.get_color(self.selected_label)
        self.events.colormap()  # Will update the LabelVispyColormap shader
        self.refresh()
        self.events.selected_label()

    @property
    def seed_rng(self) -> Optional[int]:
        return self._seed_rng

    @seed_rng.setter
    def seed_rng(self, seed_rng: Optional[int]) -> None:
        if seed_rng == self._seed_rng:
            return
        self._seed_rng = seed_rng

        if self._seed_rng is None:
            self.colormap = label_colormap(
                self.num_colors, self.seed, self._background_label
            )
        else:
            self.colormap.shuffle(self._seed_rng)
        self._cached_labels = None  # invalidate the cached color mapping
        self._selected_color = self.get_color(self.selected_label)
        self.events.colormap()  # Will update the LabelVispyColormap shader
        self.events.selected_label()

        self.refresh()

    @_ImageBase.colormap.setter
    def colormap(self, colormap):
        super()._set_colormap(colormap)
        self._selected_color = self.get_color(self.selected_label)

    @property
    def num_colors(self):
        """int: Number of unique colors to use in colormap."""
        return self._num_colors

    @num_colors.setter
    def num_colors(self, num_colors):
        self._num_colors = num_colors
        self.colormap = label_colormap(
            num_colors, self.seed, self._background_label
        )
        self.refresh()
        self._selected_color = self.get_color(self.selected_label)
        self.events.selected_label()

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        data = self._ensure_int_labels(data)
        self._data = data
        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: Union[Dict[str, np.ndarray], pd.DataFrame],
    ) -> None:
        self._feature_table.set_values(features)
        self._label_index = self._make_label_index()
        self.events.properties()
        self.events.features()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: array (N,)}, DataFrame: Properties for each label."""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: Dict[str, Array]):
        self.features = properties

    def _make_label_index(self) -> Dict[int, int]:
        features = self._feature_table.values
        label_index = {}
        if 'index' in features:
            label_index = {i: k for k, i in enumerate(features['index'])}
        elif features.shape[1] > 0:
            label_index = {i: i for i in range(features.shape[0])}
        return label_index

    @property
    def color(self):
        """dict: custom color dict for label coloring"""
        return self._color

    @color.setter
    def color(self, color):
        if not color:
            color = {}

        if self._background_label not in color:
            color[self._background_label] = 'transparent'

        default_color = color.pop(None, 'black')
        # this is default color for label that is not in the color dict
        # is provided as None key
        # we pop it as `None` cannot be cast to float
        self._validate_colors(color)

        color[None] = default_color

        colors = {
            label: transform_color(color_str)[0]
            for label, color_str in color.items()
        }
        self._color = colors
        self._direct_colormap = direct_colormap(colors)

        # `colors` may contain just the default None and background label
        # colors, in which case we need to be in AUTO color mode. Otherwise,
        # `colors` contains colors for all labels, and we should be in DIRECT
        # mode.

        # For more information
        # - https://github.com/napari/napari/issues/2479
        # - https://github.com/napari/napari/issues/2953
        if self._is_default_colors(colors):
            color_mode = LabelColorMode.AUTO
        else:
            color_mode = LabelColorMode.DIRECT

        self.color_mode = color_mode

    @classmethod
    def _validate_colors(cls, labels: Iterable[int]):
        """Check whether any of the given labels will be aliased together.

        See https://github.com/napari/napari/issues/6084 for details.
        """
        labels_int = np.fromiter(labels, dtype=int)
        labels_unique = np.unique(cls._to_vispy_texture_dtype(labels_int))
        if labels_unique.size == labels_int.size:
            return

        # recalculate here second time to provide best performance on colors that are not colliding
        labels_unique, inverse, count = np.unique(
            cls._to_vispy_texture_dtype(labels_int),
            return_inverse=True,
            return_counts=True,
        )
        collided_idx = np.where(count > 1)[0]
        aliased_list = [
            labels_int[np.where(inverse == idx)[0]] for idx in collided_idx
        ]

        alias_string = "\n".join(
            trans._(
                'Labels {col_li} will display as the same color as {col_la};',
                col_li=",".join(str(i) for i in lst[:-1]),
                col_la=str(lst[-1]),
            )
            for lst in aliased_list
        )
        warn_text = trans._(
            "Because integer labels are cast to less-precise float for display, "
            "the following label sets will render as the same color:\n"
            "{alias_string}\n"
            "See https://github.com/napari/napari/issues/6084 for details.",
            alias_string=alias_string,
        )
        warnings.warn(warn_text, category=RuntimeWarning)

    def _is_default_colors(self, color):
        """Returns True if color contains only default colors, otherwise False.

        Default colors are black for `None` and transparent for
        `self._background_label`.

        Parameters
        ----------
        color : Dict
            Dictionary of label value to color array

        Returns
        -------
        bool
            True if color contains only default colors, otherwise False.
        """
        if len(color) != 2:
            return False

        if not hasattr(self, '_color'):
            return False

        default_keys = [None, self._background_label]
        if set(default_keys) != set(color.keys()):
            return False

        for key in default_keys:
            if not np.allclose(self._color[key], color[key]):
                return False

        return True

    def _ensure_int_labels(self, data):
        """Ensure data is integer by converting from bool if required, raising an error otherwise."""
        looks_multiscale, data = guess_multiscale(data)
        if not looks_multiscale:
            data = [data]
        int_data = []
        for data_level in data:
            # normalize_dtype turns e.g. tensorstore or torch dtypes into
            # numpy dtypes
            if np.issubdtype(normalize_dtype(data_level.dtype), np.floating):
                raise TypeError(
                    trans._(
                        "Only integer types are supported for Labels layers, but data contains {data_level_type}.",
                        data_level_type=data_level.dtype,
                    )
                )
            if data_level.dtype == bool:
                int_data.append(data_level.astype(np.int8))
            else:
                int_data.append(data_level)
        data = int_data
        if not looks_multiscale:
            data = data[0]
        return data

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
                'multiscale': self.multiscale,
                'num_colors': self.num_colors,
                'properties': self.properties,
                'rendering': self.rendering,
                'depiction': self.depiction,
                'plane': self.plane.dict(),
                'experimental_clipping_planes': [
                    plane.dict() for plane in self.experimental_clipping_planes
                ],
                'seed_rng': self.seed_rng,
                'data': self.data,
                'color': self.color,
                'features': self.features,
            }
        )
        return state

    @property
    def selected_label(self):
        """int: Index of selected label."""
        return self._selected_label

    @selected_label.setter
    def selected_label(self, selected_label):
        if selected_label == self.selected_label:
            return

        self.colormap.selection = selected_label
        self._selected_label = selected_label
        self._selected_color = self.get_color(selected_label)

        self.events.selected_label()

        if self.show_selected_label:
            self._cached_labels = None  # invalidates labels cache
            self.refresh()

    @property
    def color_mode(self):
        """Color mode to change how color is represented.

        AUTO (default) allows color to be set via a hash function with a seed.

        DIRECT allows color of each label to be set directly by a color dict.
        """
        return str(self._color_mode)

    @color_mode.setter
    def color_mode(self, color_mode: Union[str, LabelColorMode]):
        color_mode = LabelColorMode(color_mode)
        self._cached_labels = None  # invalidates labels cache
        self._color_mode = color_mode
        if color_mode == LabelColorMode.AUTO:
            self._colormap = ensure_colormap(self._random_colormap)
        else:
            self._colormap = ensure_colormap(self._direct_colormap)
        self._selected_color = self.get_color(self.selected_label)
        self.events.color_mode()
        self.events.colormap()  # If remove this emitting, connect shader update to color_mode
        self.events.selected_label()
        self.refresh()

    @property
    def show_selected_label(self):
        """Whether to filter displayed labels to only the selected label or not"""
        return self._show_selected_label

    @show_selected_label.setter
    def show_selected_label(self, show_selected):
        self._show_selected_label = show_selected
        self.colormap.use_selection = show_selected
        self.colormap.selection = self.selected_label
        self.events.show_selected_label(show_selected_label=show_selected)
        self._cached_labels = None
        self.refresh()

    @Layer.mode.getter
    def mode(self):
        """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICK mode the cursor functions like a color picker, setting the
        clicked on label to be the current label. If the background is picked it
        will select the background label `0`.

        In PAINT mode the cursor functions like a paint brush changing any
        pixels it brushes over to the current label. If the background label
        `0` is selected than any pixels will be changed to background and this
        tool functions like an eraser. The size and shape of the cursor can be
        adjusted in the properties widget.

        In FILL mode the cursor functions like a fill bucket replacing pixels
        of the label clicked on with the current label. It can either replace
        all pixels of that label or just those that are contiguous with the
        clicked on pixel. If the background label `0` is selected than any
        pixels will be changed to background and this tool functions like an
        eraser.

        In ERASE mode the cursor functions similarly to PAINT mode, but to
        paint with background label, which effectively removes the label.
        """
        return str(self._mode)

    def _mode_setter_helper(self, mode):
        mode = super()._mode_setter_helper(mode)
        if mode == self._mode:
            return mode

        if mode in {Mode.PAINT, Mode.ERASE}:
            self.cursor_size = self._calculate_cursor_size()

        return mode

    @property
    def preserve_labels(self):
        """Defines if painting should preserve existing labels.

        Default to false to allow paint on existing labels. When
        set to true, existing labels will be preserved during painting.
        """
        return self._preserve_labels

    @preserve_labels.setter
    def preserve_labels(self, preserve_labels: bool):
        self._preserve_labels = preserve_labels
        self.events.preserve_labels(preserve_labels=preserve_labels)

    @property
    def contrast_limits(self):
        return self._contrast_limits

    @contrast_limits.setter
    def contrast_limits(self, value):
        # Setting contrast_limits of labels layers leads to wrong visualization
        # of the layer, so we ignore the value
        self._contrast_limits = (0, 1)

    def _reset_editable(self) -> None:
        self.editable = not self.multiscale

    def _on_editable_changed(self) -> None:
        if not self.editable:
            self.mode = Mode.PAN_ZOOM
            self._reset_history()

    @staticmethod
    def _to_vispy_texture_dtype(data):
        """Convert data to a dtype that can be used as a VisPy texture.

        Labels layers allow all integer dtypes for data, but only a subset
        are supported by VisPy textures. For now, we convert all data to
        float32 as it can represent all input values (though not losslessly,
        see https://github.com/napari/napari/issues/6084).
        """
        return vispy_texture_dtype(data)

    def _partial_labels_refresh(self):
        """Prepares and displays only an updated part of the labels."""

        if self._updated_slice is None or not self.loaded:
            return

        dims_displayed = self._slice_input.displayed
        raw_displayed = self._slice.image.raw

        # Keep only the dimensions that correspond to the current view
        updated_slice = tuple(
            self._updated_slice[index] for index in dims_displayed
        )

        offset = [axis_slice.start for axis_slice in updated_slice]

        colors_sliced = self._raw_to_displayed(
            raw_displayed, data_slice=updated_slice
        )

        self.events.labels_update(data=colors_sliced, offset=offset)
        self._updated_slice = None

    def _raw_to_displayed(self, raw, data_slice: Tuple[slice] = None):
        """Determine displayed image from a saved raw image and a saved seed.

        This function ensures that the 0 label gets mapped to the 0 displayed
        pixel.

        Parameters
        ----------
        raw : array or int
            Raw integer input image.

        data_slice : numpy array slice
            Slice that specifies the portion of the input image that
            should be computed and displayed.
            If None, the whole input image will be processed.
        Returns
        -------
        mapped_labels : array
            Encoded colors mapped between 0 and 1 to be displayed.
        """
        if data_slice is None:
            data_slice = tuple(slice(0, size) for size in raw.shape)

        labels = raw  # for readability
        sliced_labels = None

        # lookup function -> self._as_type
        if self.contour > 0:
            if labels.ndim == 2:
                # Add one more pixel for the correct borders computation
                expanded_slice = expand_slice(data_slice, labels.shape, 1)
                sliced_labels = get_contours(
                    labels[expanded_slice],
                    self.contour,
                    self._background_label,
                )

                # Remove the latest one-pixel border from the result
                delta_slice = tuple(
                    [
                        slice(s1.start - s2.start, s1.stop - s2.start)
                        for s1, s2 in zip(data_slice, expanded_slice)
                    ]
                )
                sliced_labels = sliced_labels[delta_slice]
            elif labels.ndim > 2:
                warnings.warn(
                    trans._(
                        "Contours are not displayed during 3D rendering",
                        deferred=True,
                    )
                )

        if sliced_labels is None:
            sliced_labels = labels[data_slice]

        # cache the labels and keep track of when values are changed
        update_mask = None
        if (
            self._cached_labels is not None
            and self._cached_labels.shape == labels.shape
        ):
            update_mask = self._cached_labels[data_slice] != sliced_labels
            # Select only a subset with changes for further computations
            labels_to_map = sliced_labels[update_mask]
            # Update the cache
            self._cached_labels[data_slice][update_mask] = labels_to_map
        else:
            self._cached_labels = np.zeros_like(labels)
            self._cached_mapped_labels = np.zeros_like(
                labels, dtype=np.float32
            )
            self._cached_labels[data_slice] = sliced_labels.copy()
            labels_to_map = sliced_labels

        # If there are no changes, just return the cached image
        if labels_to_map.size == 0:
            return self._cached_mapped_labels[data_slice]

        mapped_labels = self._to_vispy_texture_dtype(labels_to_map)

        if update_mask is not None:
            self._cached_mapped_labels[data_slice][update_mask] = mapped_labels
        else:
            self._cached_mapped_labels[data_slice] = mapped_labels

        return self._cached_mapped_labels[data_slice]

    def _update_thumbnail(self):
        """Update the thumbnail with current data and colormap.

        This is overridden from _ImageBase because we don't need to do things
        like adjusting gamma or changing the data based on the contrast
        limits.
        """
        if not self.loaded:
            # ASYNC_TODO: Do not compute the thumbnail until we are loaded.
            # Is there a nicer way to prevent this from getting called?
            return

        image = self._slice.thumbnail.view
        if self._slice_input.ndisplay == 3 and self.ndim > 2:
            # we are only using the current slice so `image` will never be
            # bigger than 3. If we are in this clause, it is exactly 3, so we
            # use max projection. For labels, ideally we would use "first
            # nonzero projection", but we leave that for a future PR. (TODO)
            image = np.max(image, axis=0)
        imshape = np.array(image.shape[:2])
        thumbshape = np.array(self._thumbnail_shape[:2])

        raw_zoom_factor = np.min(thumbshape / imshape)
        new_shape = np.clip(
            raw_zoom_factor * imshape, a_min=1, a_max=thumbshape
        )
        zoom_factor = tuple(new_shape / imshape)

        downsampled = ndi.zoom(image, zoom_factor, prefilter=False, order=0)
        if self.color_mode == LabelColorMode.AUTO:
            color_array = self.colormap.map(downsampled.ravel())
        else:  # direct
            color_array = self._direct_colormap.map(downsampled.ravel())
        colormapped = color_array.reshape(downsampled.shape + (4,))
        colormapped[..., 3] *= self.opacity

        self.thumbnail = colormapped

    def new_colormap(self):
        self.seed_rng = np.random.default_rng().integers(2**32 - 1)

    def get_color(self, label):
        """Return the color corresponding to a specific label."""
        if label == self._background_label:
            col = None
        elif label is None or (
            self.show_selected_label and label != self.selected_label
        ):
            col = self.colormap.map([0, 0, 0, 0])[0]
        else:
            val = self._to_vispy_texture_dtype(np.array([label]))
            col = self.colormap.map(val)[0]
        return col

    def _get_value_ray(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : Optional[int]
            The first non-zero value encountered along the ray. If none
            was encountered or the viewer is in 2D mode, None is returned.
        """
        if start_point is None or end_point is None:
            return None
        if len(dims_displayed) == 3:
            # only use get_value_ray on 3D for now
            # we use dims_displayed because the image slice
            # has its dimensions  in th same order as the vispy
            # Volume
            start_point = start_point[dims_displayed]
            end_point = end_point[dims_displayed]
            sample_ray = end_point - start_point
            length_sample_vector = np.linalg.norm(sample_ray)
            n_points = int(2 * length_sample_vector)
            sample_points = np.linspace(
                start_point, end_point, n_points, endpoint=True
            )
            im_slice = self._slice.image.raw
            clamped = clamp_point_to_bounding_box(
                sample_points, self._display_bounding_box(dims_displayed)
            ).astype(int)
            values = im_slice[tuple(clamped.T)]
            nonzero_indices = np.flatnonzero(values)
            if len(nonzero_indices > 0):
                # if a nonzer0 value was found, return the first one
                return values[nonzero_indices[0]]

        return None

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : int
            The first non-zero value encountered along the ray. If a
            non-zero value is not encountered, returns 0 (the background value).
        """
        return (
            self._get_value_ray(
                start_point=start_point,
                end_point=end_point,
                dims_displayed=dims_displayed,
            )
            or 0
        )

    def _reset_history(self, event=None):
        self._undo_history = deque(maxlen=self._history_limit)
        self._redo_history = deque(maxlen=self._history_limit)
        self._staged_history = []
        self._block_history = False

    @contextmanager
    def block_history(self):
        """Context manager to group history-editing operations together.

        While in the context, history atoms are grouped together into a
        "staged" history. When exiting the context, that staged history is
        committed to the undo history queue, and an event is emitted
        containing the change.
        """
        prev = self._block_history
        self._block_history = True
        try:
            yield
            self._commit_staged_history()
        finally:
            self._block_history = prev

    def _commit_staged_history(self):
        """Save staged history to undo history and clear it."""
        if self._staged_history:
            self._append_to_undo_history(self._staged_history)
            self._staged_history = []

    def _append_to_undo_history(self, item):
        """Append item to history and emit paint event.

        Parameters
        ----------
        item : List[Tuple[ndarray, ndarray, int]]
            list of history atoms to append to undo history.
        """
        self._undo_history.append(item)
        self.events.paint(value=item)

    def _save_history(self, value):
        """Save a history "atom" to the undo history.

        A history "atom" is a single change operation to the array. A history
        *item* is a collection of atoms that were applied together to make a
        single change. For example, when dragging and painting, at each mouse
        callback we create a history "atom", but we save all those atoms in
        a single history item, since we would want to undo one drag in one
        undo operation.

        Parameters
        ----------
        value : 3-tuple of arrays
            The value is a 3-tuple containing:

            - a numpy multi-index, pointing to the array elements that were
              changed
            - the values corresponding to those elements before the change
            - the value(s) after the change
        """
        self._redo_history.clear()
        if self._block_history:
            self._staged_history.append(value)
        else:
            self._append_to_undo_history([value])

    def _load_history(self, before, after, undoing=True):
        """Load a history item and apply it to the array.

        Parameters
        ----------
        before : list of history items
            The list of elements from which we want to load.
        after : list of history items
            The list of element to which to append the loaded element. In the
            case of an undo operation, this is the redo queue, and vice versa.
        undoing : bool
            Whether we are undoing (default) or redoing. In the case of
            redoing, we apply the "after change" element of a history element
            (the third element of the history "atom").

        See Also
        --------
        Labels._save_history
        """
        if len(before) == 0:
            return

        history_item = before.pop()
        after.append(list(reversed(history_item)))
        for prev_indices, prev_values, next_values in reversed(history_item):
            values = prev_values if undoing else next_values
            self.data[prev_indices] = values

        self.refresh()

    def undo(self):
        self._load_history(
            self._undo_history, self._redo_history, undoing=True
        )

    def redo(self):
        self._load_history(
            self._redo_history, self._undo_history, undoing=False
        )

    def fill(self, coord, new_label, refresh=True):
        """Replace an existing label with a new label, either just at the
        connected component if the `contiguous` flag is `True` or everywhere
        if it is `False`, working in the number of dimensions specified by
        the `n_edit_dimensions` flag.

        Parameters
        ----------
        coord : sequence of float
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        int_coord = tuple(np.round(coord).astype(int))
        # If requested fill location is outside data shape then return
        if np.any(np.less(int_coord, 0)) or np.any(
            np.greater_equal(int_coord, self.data.shape)
        ):
            return

        # If requested new label doesn't change old label then return
        old_label = np.asarray(self.data[int_coord]).item()
        if old_label == new_label or (
            self.preserve_labels and old_label != self._background_label
        ):
            return

        dims_to_fill = sorted(
            self._slice_input.order[-self.n_edit_dimensions :]
        )
        data_slice_list = list(int_coord)
        for dim in dims_to_fill:
            data_slice_list[dim] = slice(None)
        data_slice = tuple(data_slice_list)
        labels = np.asarray(self.data[data_slice])
        slice_coord = tuple(int_coord[d] for d in dims_to_fill)

        matches = labels == old_label
        if self.contiguous:
            # if contiguous replace only selected connected component
            labeled_matches, num_features = ndi.label(matches)
            if num_features != 1:
                match_label = labeled_matches[slice_coord]
                matches = np.logical_and(
                    matches, labeled_matches == match_label
                )

        match_indices_local = np.nonzero(matches)
        if self.ndim not in {2, self.n_edit_dimensions}:
            n_idx = len(match_indices_local[0])
            match_indices = []
            j = 0
            for d in data_slice:
                if isinstance(d, slice):
                    match_indices.append(match_indices_local[j])
                    j += 1
                else:
                    match_indices.append(np.full(n_idx, d, dtype=np.intp))
        else:
            match_indices = match_indices_local

        match_indices = _coerce_indices_for_vectorization(
            self.data, match_indices
        )

        self.data_setitem(match_indices, new_label, refresh)

    def _draw(self, new_label, last_cursor_coord, coordinates):
        """Paint into coordinates, accounting for mode and cursor movement.

        The draw operation depends on the current mode of the layer.

        Parameters
        ----------
        new_label : int
            value of label to paint
        last_cursor_coord : sequence
            last painted cursor coordinates
        coordinates : sequence
            new cursor coordinates
        """
        if coordinates is None:
            return
        interp_coord = interpolate_coordinates(
            last_cursor_coord, coordinates, self.brush_size
        )
        for c in interp_coord:
            if (
                self._slice_input.ndisplay == 3
                and self.data[tuple(np.round(c).astype(int))] == 0
            ):
                continue
            if self._mode in [Mode.PAINT, Mode.ERASE]:
                self.paint(c, new_label, refresh=False)
            elif self._mode == Mode.FILL:
                self.fill(c, new_label, refresh=False)
        self._partial_labels_refresh()

    def paint(self, coord, new_label, refresh=True):
        """Paint over existing labels with a new label, using the selected
        brush shape and size, either only on the visible slice or in all
        n dimensions.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        shape = self.data.shape
        dims_to_paint = sorted(
            self._slice_input.order[-self.n_edit_dimensions :]
        )
        dims_not_painted = sorted(
            self._slice_input.order[: -self.n_edit_dimensions]
        )
        paint_scale = np.array(
            [self.scale[i] for i in dims_to_paint], dtype=float
        )

        slice_coord = [int(np.round(c)) for c in coord]
        if self.n_edit_dimensions < self.ndim:
            coord_paint = [coord[i] for i in dims_to_paint]
            shape = [shape[i] for i in dims_to_paint]
        else:
            coord_paint = coord

        # Ensure circle doesn't have spurious point
        # on edge by keeping radius as ##.5
        radius = np.floor(self.brush_size / 2) + 0.5
        mask_indices = sphere_indices(radius, tuple(paint_scale))

        mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
            int
        )

        # discard candidate coordinates that are out of bounds
        mask_indices = indices_in_shape(mask_indices, shape)

        # Transfer valid coordinates to slice_coord,
        # or expand coordinate if 3rd dim in 2D image
        slice_coord_temp = list(mask_indices.T)
        if self.n_edit_dimensions < self.ndim:
            for j, i in enumerate(dims_to_paint):
                slice_coord[i] = slice_coord_temp[j]
            for i in dims_not_painted:
                slice_coord[i] = slice_coord[i] * np.ones(
                    mask_indices.shape[0], dtype=int
                )
        else:
            slice_coord = slice_coord_temp

        slice_coord = _coerce_indices_for_vectorization(self.data, slice_coord)

        # slice coord is a tuple of coordinate arrays per dimension
        # subset it if we want to only paint into background/only erase
        # current label
        if self.preserve_labels:
            if new_label == self._background_label:
                keep_coords = self.data[slice_coord] == self.selected_label
            else:
                keep_coords = self.data[slice_coord] == self._background_label
            slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

        self.data_setitem(slice_coord, new_label, refresh)

    def data_setitem(self, indices, value, refresh=True):
        """Set `indices` in `data` to `value`, while writing to edit history.

        Parameters
        ----------
        indices : tuple of arrays of int
            Indices in data to overwrite. Must be a tuple of arrays of length
            equal to the number of data dimensions. (Fancy indexing in [1]_).
        value : int or array of int
            New label value(s). If more than one value, must match or
            broadcast with the given indices.
        refresh : bool, default True
            whether to refresh the view, by default True

        References
        ----------
        ..[1] https://numpy.org/doc/stable/user/basics.indexing.html
        """
        changed_indices = self.data[indices] != value
        indices = tuple([x[changed_indices] for x in indices])

        if not indices or indices[0].size == 0:
            return

        self._save_history(
            (
                indices,
                np.array(self.data[indices], copy=True),
                value,
            )
        )

        # update the labels image
        self.data[indices] = value

        if not (  # if not a numpy array or numpy-backed xarray
            isinstance(self.data, np.ndarray)
            or isinstance(getattr(self.data, 'data', None), np.ndarray)
        ):
            # In the absence of slicing, the current slice becomes
            # invalidated by data_setitem; only in the special case of a NumPy
            # array, or a NumPy-array-backed Xarray, is the slice a view and
            # therefore updated automatically.
            # For other types, we update it manually here.

            point = np.round(
                self.world_to_data(self._slice_input.point)
            ).astype(int)
            pt_not_disp = {
                dim: point[dim] for dim in self._slice_input.not_displayed
            }
            displayed_indices = index_in_slice(indices, pt_not_disp)
            self._slice.image.raw[displayed_indices] = value

        # tensorstore and xarray do not return their indices in
        # np.ndarray format, so they need to be converted explicitly
        if not isinstance(self.data, np.ndarray):
            indices = [np.array(x).flatten() for x in indices]

        updated_slice = tuple(
            [
                slice(min(axis_indices), max(axis_indices) + 1)
                for axis_indices in indices
            ]
        )

        if self.contour > 0:
            # Expand the slice by 1 pixel as the changes can go beyond
            # the original slice because of the morphological dilation
            # (1 pixel because get_countours always applies 1 pixel dilation)
            updated_slice = expand_slice(updated_slice, self.data.shape, 1)

        if self._updated_slice is None:
            self._updated_slice = updated_slice
        else:
            self._updated_slice = tuple(
                [
                    slice(min(s1.start, s2.start), max(s1.stop, s2.stop))
                    for s1, s2 in zip(updated_slice, self._updated_slice)
                ]
            )

        if refresh is True:
            self._partial_labels_refresh()

    def get_status(
        self,
        position: Optional[Tuple] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ) -> dict:
        """Status message information of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        source_info : dict
            Dict containing a information that can be used in a status update.
        """
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = self._get_source_info()
        source_info['coordinates'] = generate_layer_coords_status(
            position[-self.ndim :], value
        )

        # if this labels layer has properties
        properties = self._get_properties(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if properties:
            source_info['coordinates'] += "; " + ", ".join(properties)

        return source_info

    def _get_tooltip_text(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ):
        """
        tooltip message of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        msg : string
            String containing a message that can be used as a tooltip.
        """
        return "\n".join(
            self._get_properties(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        )

    def _get_properties(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ) -> list:
        if len(self._label_index) == 0 or self.features.shape[1] == 0:
            return []

        value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        # if the cursor is not outside the image or on the background
        if value is None:
            return []

        label_value = value[1] if self.multiscale else value
        if label_value not in self._label_index:
            return [trans._('[No Properties]')]

        idx = self._label_index[label_value]
        return [
            f'{k}: {v[idx]}'
            for k, v in self.features.items()
            if k != 'index'
            and len(v) > idx
            and v[idx] is not None
            and not (isinstance(v[idx], float) and np.isnan(v[idx]))
        ]


if config.async_octree:
    from napari.layers.image.experimental.octree_image import _OctreeImageBase

    class Labels(Labels, _OctreeImageBase):
        pass


def _coerce_indices_for_vectorization(array, indices: list) -> tuple:
    """Coerces indices so that they can be used for vectorized indexing in the given data array."""
    if _is_array_type(array, 'xarray.DataArray'):
        # Fix indexing for xarray if necessary
        # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
        # for difference from indexing numpy
        try:
            import xarray as xr
        except ModuleNotFoundError:
            pass
        else:
            return tuple(xr.DataArray(i) for i in indices)
    return tuple(indices)
