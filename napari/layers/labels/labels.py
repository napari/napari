import warnings
from collections import deque
from typing import Dict, Union

import numpy as np
from scipy import ndimage as ndi

from ...utils.colormaps import (
    color_dict_to_colormap,
    label_colormap,
    low_discrepancy_image,
)
from ...utils.events import Event
from ..image.image import _ImageBase
from ..utils.color_transformations import transform_color
from ..utils.layer_utils import dataframe_to_properties
from ._labels_constants import LabelBrushShape, LabelColorMode, Mode
from ._labels_mouse_bindings import draw, pick
from ._labels_utils import sphere_indices


class Labels(_ImageBase):
    """Labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or multiscale.
    num_colors : int
        Number of unique colors to use in colormap.
    properties : dict {str: array (N,)}, DataFrame
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
        the final column is a lenght N translation vector and a 1 or a napari
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
    data : array
        Integer valued label data. Can be N dimensional. Every pixel contains
        an integer ID corresponding to the region it belongs to. The label 0 is
        rendered as transparent.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays.
    seed : float
        Seed for colormap random generator.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    contiguous : bool
        If `True`, the fill bucket changes only connected pixels of same label.
    n_dimensional : bool
        If `True`, paint and fill edit labels across all dimensions.
    contour : int
        If greater than 0, displays contours of labels instead of shaded regions
        with a thickness equal to its value.
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

    Extended Summary
    ----------
    _data_raw : array (N, M)
        2D labels data for the currently viewed slice.
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    """

    _history_limit = 100

    def __init__(
        self,
        data,
        *,
        num_colors=50,
        properties=None,
        color=None,
        seed=0.5,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending='translucent',
        visible=True,
        multiscale=None,
    ):

        self._seed = seed
        self._background_label = 0
        self._num_colors = num_colors
        self._random_colormap = label_colormap(self.num_colors)
        self._color_mode = LabelColorMode.AUTO
        self._brush_shape = LabelBrushShape.CIRCLE
        self._show_selected_label = False
        self._contour = 0

        if properties is None:
            self._properties = {}
            label_index = {}
        else:
            properties = self._validate_properties(properties)
            self._properties, label_index = dataframe_to_properties(properties)
        if label_index is None:
            props = self._properties
            if len(props) > 0:
                self._label_index = self._map_index(properties)
            else:
                self._label_index = {}
        else:
            self._label_index = label_index

        super().__init__(
            data,
            rgb=False,
            colormap=self._random_colormap,
            contrast_limits=[0.0, 1.0],
            interpolation='nearest',
            rendering='translucent',
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
            mode=Event,
            preserve_labels=Event,
            properties=Event,
            n_dimensional=Event,
            contiguous=Event,
            brush_size=Event,
            selected_label=Event,
            color_mode=Event,
            brush_shape=Event,
            contour=Event,
        )

        self._n_dimensional = False
        self._contiguous = True
        self._brush_size = 10

        self._selected_label = 1
        self._selected_color = self.get_color(self._selected_label)
        self.color = color

        self._mode = Mode.PAN_ZOOM
        self._mode_history = self._mode
        self._status = self.mode
        self._preserve_labels = False
        self._help = 'enter paint or fill mode to edit labels'

        self._block_saving = False
        self._reset_history()

        # Trigger generation of view slice and thumbnail
        self._update_dims()
        self._set_editable()

    @property
    def contiguous(self):
        """bool: fill bucket changes only connected pixels of same label."""
        return self._contiguous

    @contiguous.setter
    def contiguous(self, contiguous):
        self._contiguous = contiguous
        self.events.contiguous()

    @property
    def n_dimensional(self):
        """bool: paint and fill edits labels across all dimensions."""
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional):
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()

    @property
    def contour(self):
        """int: displays contours of labels instead of shaded regions."""
        return self._contour

    @contour.setter
    def contour(self, contour):
        self._contour = contour
        self.events.contour()
        self.refresh()

    @property
    def brush_size(self):
        """float: Size of the paint in world coordinates."""
        return self._brush_size

    @brush_size.setter
    def brush_size(self, brush_size):
        self._brush_size = int(brush_size)
        # Convert from brush size in data coordinates to
        # cursor size in world coordinates
        data2world_scale = np.mean(
            [self.scale[d] for d in self._dims_displayed]
        )
        self.cursor_size = self.brush_size * data2world_scale
        self.events.brush_size()

    @property
    def seed(self):
        """float: Seed for colormap random generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._selected_color = self.get_color(self.selected_label)
        self.refresh()
        self.events.selected_label()

    @property
    def num_colors(self):
        """int: Number of unique colors to use in colormap."""
        return self._num_colors

    @num_colors.setter
    def num_colors(self, num_colors):
        self._num_colors = num_colors
        self.colormap = label_colormap(num_colors)
        self.refresh()
        self._selected_color = self.get_color(self.selected_label)
        self.events.selected_label()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: array (N,)}, DataFrame: Properties for each label."""
        return self._properties

    @properties.setter
    def properties(self, properties: Dict[str, np.ndarray]):
        if not isinstance(properties, dict):
            properties, label_index = dataframe_to_properties(properties)
            if label_index is None:
                label_index = self._map_index(properties)
        self._properties = self._validate_properties(properties)
        self._label_index = label_index
        self.events.properties()

    @property
    def color(self):
        """dict: custom color dict for label coloring"""
        return self._color

    @color.setter
    def color(self, color):

        if not color:
            color = {}
            color_mode = LabelColorMode.AUTO
        else:
            color_mode = LabelColorMode.DIRECT

        if self._background_label not in color:
            color[self._background_label] = 'transparent'

        if None not in color:
            color[None] = 'black'

        colors = {
            label: transform_color(color_str)[0]
            for label, color_str in color.items()
        }

        self._color = colors
        self.color_mode = color_mode

    def _validate_properties(
        self, properties: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Validate the type and size of properties."""
        lens = []
        for k, v in properties.items():
            lens.append(len(v))
            if not isinstance(v, np.ndarray):
                properties[k] = np.asarray(v)

        if not all([v == lens[0] for v in lens]):
            raise ValueError(
                "the number of items must be equal for all properties"
            )
        return properties

    def _map_index(self, properties: Dict[str, np.ndarray]) -> Dict[int, int]:
        """Map rows in given properties to label indices"""
        arbitrary_key = list(properties.keys())[0]
        label_index = {i: i for i in range(len(properties[arbitrary_key]))}
        return label_index

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
                'properties': self._properties,
                'seed': self.seed,
                'data': self.data,
                'color': self.color,
            }
        )
        return state

    @property
    def selected_label(self):
        """int: Index of selected label."""
        return self._selected_label

    @selected_label.setter
    def selected_label(self, selected_label):
        if selected_label < 0:
            raise ValueError('cannot reduce selected label below 0')
        if selected_label == self.selected_label:
            return

        self._selected_label = selected_label
        self._selected_color = self.get_color(selected_label)
        self.events.selected_label()

        # note: self.color_mode returns a string and this comparison fails,
        # so use self._color_mode
        if self.show_selected_label:
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
        if color_mode == LabelColorMode.DIRECT:
            (
                custom_colormap,
                label_color_index,
            ) = color_dict_to_colormap(self.color)
            self.colormap = custom_colormap
            self._label_color_index = label_color_index
        elif color_mode == LabelColorMode.AUTO:
            self._label_color_index = {}
            self.colormap = self._random_colormap

        else:
            raise ValueError("Unsupported Color Mode")

        self._color_mode = color_mode
        self._selected_color = self.get_color(self.selected_label)
        self.events.color_mode()
        self.events.colormap()
        self.events.selected_label()
        self.refresh()

    @property
    def show_selected_label(self):
        """Whether to filter displayed labels to only the selected label or not"""
        return self._show_selected_label

    @show_selected_label.setter
    def show_selected_label(self, filter):
        self._show_selected_label = filter
        self.refresh()

    @property
    def brush_shape(self):
        """str: Paintbrush shape"""
        return str(self._brush_shape)

    @brush_shape.setter
    def brush_shape(self, brush_shape):
        """Set current brush shape."""
        self._brush_shape = LabelBrushShape(brush_shape)
        self.cursor = self.brush_shape

    @property
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

    @mode.setter
    def mode(self, mode: Union[str, Mode]):
        mode = Mode(mode)

        if not self.editable:
            mode = Mode.PAN_ZOOM

        if mode == self._mode:
            return

        if self._mode == Mode.PICK:
            self.mouse_drag_callbacks.remove(pick)
        elif self._mode in [Mode.PAINT, Mode.FILL, Mode.ERASE]:
            self.mouse_drag_callbacks.remove(draw)

        if mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = 'enter paint or fill mode to edit labels'
        elif mode == Mode.PICK:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, click to pick a label'
            self.mouse_drag_callbacks.append(pick)
        elif mode == Mode.PAINT:
            self.cursor = self.brush_shape
            # Convert from brush size in data coordinates to
            # cursor size in world coordinates
            data2world_scale = np.mean(
                [self.scale[d] for d in self._dims_displayed]
            )
            self.cursor_size = self.brush_size * data2world_scale
            self.interactive = False
            self.help = (
                'hold <space> to pan/zoom, '
                'hold <shift> to toggle preserve_labels, '
                'hold <control> to fill, '
                'hold <alt> to erase, '
                'drag to paint a label'
            )
            self.mouse_drag_callbacks.append(draw)
        elif mode == Mode.FILL:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, click to fill a label'
            self.mouse_drag_callbacks.append(draw)
        elif mode == Mode.ERASE:
            self.cursor = self.brush_shape
            # Convert from brush size in data coordinates to
            # cursor size in world coordinates
            data2world_scale = np.mean(
                [self.scale[d] for d in self._dims_displayed]
            )
            self.cursor_size = self.brush_size * data2world_scale
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, drag to erase a label'
            self.mouse_drag_callbacks.append(draw)
        else:
            raise ValueError("Mode not recognized")

        self._mode = mode

        self.events.mode(mode=mode)
        self.refresh()

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

    def _set_editable(self, editable=None):
        """Set editable mode based on layer properties."""
        if editable is None:
            if self.multiscale or self._ndisplay == 3:
                self.editable = False
            else:
                self.editable = True

        if not self.editable:
            self.mode = Mode.PAN_ZOOM
            self._reset_history()

    def _raw_to_displayed(self, raw):
        """Determine displayed image from a saved raw image and a saved seed.

        This function ensures that the 0 label gets mapped to the 0 displayed
        pixel.

        Parameters
        ----------
        raw : array or int
            Raw integer input image.

        Returns
        -------
        image : array
            Image mapped between 0 and 1 to be displayed.
        """
        if (
            not self.show_selected_label
            and self._color_mode == LabelColorMode.DIRECT
        ):
            u, inv = np.unique(raw, return_inverse=True)
            image = np.array(
                [
                    self._label_color_index[x]
                    if x in self._label_color_index
                    else self._label_color_index[None]
                    for x in u
                ]
            )[inv].reshape(raw.shape)
        elif (
            not self.show_selected_label
            and self._color_mode == LabelColorMode.AUTO
        ):
            image = np.where(
                raw > 0, low_discrepancy_image(raw, self._seed), 0
            )
        elif (
            self.show_selected_label
            and self._color_mode == LabelColorMode.AUTO
        ):
            selected = self._selected_label
            image = np.where(
                raw == selected,
                low_discrepancy_image(selected, self._seed),
                0,
            )
        elif (
            self.show_selected_label
            and self._color_mode == LabelColorMode.DIRECT
        ):
            selected = self._selected_label
            if selected not in self._label_color_index:
                selected = None
            index = self._label_color_index
            image = np.where(
                raw == selected,
                index[selected],
                np.where(
                    raw != self._background_label,
                    index[None],
                    index[self._background_label],
                ),
            )
        else:
            raise ValueError("Unsupported Color Mode")

        if self.contour > 0 and raw.ndim == 2:
            image = np.zeros_like(raw)
            struct_elem = ndi.generate_binary_structure(raw.ndim, 1)
            thickness = self.contour
            thick_struct_elem = ndi.iterate_structure(
                struct_elem, thickness
            ).astype(bool)
            boundaries = ndi.grey_dilation(
                raw, footprint=struct_elem
            ) != ndi.grey_erosion(raw, footprint=thick_struct_elem)
            image[boundaries] = raw[boundaries]
            image = np.where(
                image > 0, low_discrepancy_image(image, self._seed), 0
            )
        elif self.contour > 0 and raw.ndim > 2:
            warnings.warn("Contours are not displayed during 3D rendering")

        return image

    def new_colormap(self):
        self.seed = np.random.rand()

    def get_color(self, label):
        """Return the color corresponding to a specific label."""
        if label == 0:
            col = None
        else:
            val = self._raw_to_displayed(np.array([label]))
            col = self.colormap.map(val)[0]
        return col

    def _reset_history(self, event=None):
        self._undo_history = deque()
        self._redo_history = deque()

    def _trim_history(self):
        while (
            len(self._undo_history) + len(self._redo_history)
            > self._history_limit
        ):
            self._undo_history.popleft()

    def _save_history(self):
        self._redo_history = deque()
        if not self._block_saving:
            self._undo_history.append(self.data[self._slice_indices].copy())
            self._trim_history()

    def _load_history(self, before, after):
        if len(before) == 0:
            return

        prev = before.pop()
        after.append(self.data[self._slice_indices].copy())
        self.data[self._slice_indices] = prev

        self.refresh()

    def undo(self):
        self._load_history(self._undo_history, self._redo_history)

    def redo(self):
        self._load_history(self._redo_history, self._undo_history)

    def fill(self, coord, new_label, refresh=True):
        """Replace an existing label with a new label, either just at the
        connected component if the `contiguous` flag is `True` or everywhere
        if it is `False`, working either just in the current slice if
        the `n_dimensional` flag is `False` or on the entire data if it is
        `True`.

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
        old_label = self.data[int_coord]
        if old_label == new_label or (
            self.preserve_labels and old_label != self._background_label
        ):
            return

        if refresh is True:
            self._save_history()

        if self.n_dimensional or self.ndim == 2:
            # work with entire image
            labels = self.data
            slice_coord = tuple(int_coord)
        else:
            # work with just the sliced image
            labels = self._data_raw
            slice_coord = tuple(int_coord[d] for d in self._dims_displayed)

        matches = labels == old_label
        if self.contiguous:
            # if contiguous replace only selected connected component
            labeled_matches, num_features = ndi.label(matches)
            if num_features != 1:
                match_label = labeled_matches[slice_coord]
                matches = np.logical_and(
                    matches, labeled_matches == match_label
                )

        # Replace target pixels with new_label
        labels[matches] = new_label

        if not (self.n_dimensional or self.ndim == 2):
            # if working with just the slice, update the rest of the raw data
            self.data[tuple(self._slice_indices)] = labels

        if refresh is True:
            self.refresh()

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
        if refresh is True:
            self._save_history()

        if self.brush_shape == "square":
            brush_size_dims = [self.brush_size] * self.ndim
            if not self.n_dimensional and self.ndim > 2:
                for i in self._dims_not_displayed:
                    brush_size_dims[i] = 1

            slice_coord = tuple(
                slice(
                    np.round(np.clip(c - brush_size / 2 + 0.5, 0, s)).astype(
                        int
                    ),
                    np.round(np.clip(c + brush_size / 2 + 0.5, 0, s)).astype(
                        int
                    ),
                    1,
                )
                for c, s, brush_size in zip(
                    coord, self.data.shape, brush_size_dims
                )
            )
        elif self.brush_shape == "circle":
            slice_coord = [int(np.round(c)) for c in coord]
            shape = self.data.shape
            if not self.n_dimensional and self.ndim > 2:
                coord = [coord[i] for i in self._dims_displayed]
                shape = [shape[i] for i in self._dims_displayed]

            sphere_dims = len(coord)
            # Ensure circle doesn't have spurious point
            # on edge by keeping radius as ##.5
            radius = np.floor(self.brush_size / 2) + 0.5
            mask_indices = sphere_indices(radius, sphere_dims)

            mask_indices = mask_indices + np.round(np.array(coord)).astype(int)

            # discard candidate coordinates that are out of bounds
            discard_coords = np.logical_and(
                ~np.any(mask_indices < 0, axis=1),
                ~np.any(mask_indices >= np.array(shape), axis=1),
            )
            mask_indices = mask_indices[discard_coords]

            # Transfer valid coordinates to slice_coord,
            # or expand coordinate if 3rd dim in 2D image
            slice_coord_temp = [m for m in mask_indices.T]
            if not self.n_dimensional and self.ndim > 2:
                for j, i in enumerate(self._dims_displayed):
                    slice_coord[i] = slice_coord_temp[j]
                for i in self._dims_not_displayed:
                    slice_coord[i] = slice_coord[i] * np.ones(
                        mask_indices.shape[0], dtype=int
                    )
            else:
                slice_coord = slice_coord_temp

            slice_coord = tuple(slice_coord)

        # Fix indexing for xarray if necessary
        # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
        # for difference from indexing numpy
        try:
            import xarray as xr

            if isinstance(self.data, xr.DataArray):
                slice_coord = tuple(xr.DataArray(i) for i in slice_coord)
        except ImportError:
            pass

        # slice_coord from square brush is tuple of slices per dimension
        # slice_coord from circle brush is tuple of coord. arrays per dimension

        # update the labels image

        if not self.preserve_labels:
            self.data[slice_coord] = new_label
        else:
            if new_label == self._background_label:
                keep_coords = self.data[slice_coord] == self.selected_label
            else:
                keep_coords = self.data[slice_coord] == self._background_label
            if self.brush_shape == "circle":
                slice_coord = tuple(sc[keep_coords] for sc in slice_coord)
                self.data[slice_coord] = new_label
            else:
                self.data[slice_coord][keep_coords] = new_label

        if refresh is True:
            self.refresh()

    def get_status(self, position=None, world=False):
        """Status message of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        msg : string
            String containing a message that can be used as a status update.
        """
        msg = super().get_status(position, world=world)

        # if this labels layer has properties
        if self._label_index and self._properties:
            value = self.get_value(position, world=world)
            # if the cursor is not outside the image or on the background
            if value is not None:
                if self.multiscale:
                    label_value = value[1]
                else:
                    label_value = value
                if label_value in self._label_index:
                    idx = self._label_index[label_value]
                    for k, v in self._properties.items():
                        if k != 'index':
                            msg += f', {k}: {v[idx]}'
                else:
                    msg += ' [No Properties]'
        return msg
