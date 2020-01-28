from collections import deque
from copy import copy
from typing import Union

import numpy as np
from scipy import ndimage as ndi

from ..image import Image
from ...utils.colormaps import colormaps
from ...utils.event import Event
from .labels_utils import interpolate_coordinates
from ...utils.status_messages import format_float
from ._constants import Mode


class Labels(Image):
    """Labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or pyramid.
    is_pyramid : bool
        Whether the data is an image pyramid or not. Pyramid data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be a pyramid. The first image in the list
        should be the largest.
    num_colors : int
        Number of unique colors to use in colormap.
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
        Integer valued label data. Can be N dimensional. Every pixel contains
        an integer ID corresponding to the region it belongs to. The label 0 is
        rendered as transparent.
    is_pyramid : bool
        Whether the data is an image pyramid or not. Pyramid data is
        represented by a list of array like image data. The first image in the
        list should be the largest.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap.
    seed : float
        Seed for colormap random generator.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    contiguous : bool
        If `True`, the fill bucket changes only connected pixels of same label.
    n_dimensional : bool
        If `True`, paint and fill edit labels across all dimensions.
    brush_size : float
        Size of the paint brush.
    selected_label : int
        Index of selected label. Can be greater than the current maximum label.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICKER mode the cursor functions like a color picker, setting the
        clicked on label to be the curent label. If the background is picked it
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

    Extended Summary
    ----------
    _data_raw : array (N, M)
        2D labels data for the currently viewed slice.
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    _last_cursor_coord : list or None
        Coordinates of last cursor click before painting, gets reset to None
        after painting is done. Used for interpolating brush strokes.
    """

    _history_limit = 100

    def __init__(
        self,
        data,
        *,
        is_pyramid=None,
        num_colors=50,
        seed=0.5,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=0.7,
        blending='translucent',
        visible=True,
    ):

        self._seed = seed
        self._num_colors = num_colors
        colormap = ('random', colormaps.label_colormap(self.num_colors))

        super().__init__(
            data,
            rgb=False,
            is_pyramid=is_pyramid,
            colormap=colormap,
            contrast_limits=[0.0, 1.0],
            interpolation='nearest',
            rendering='translucent',
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self.events.add(
            mode=Event,
            n_dimensional=Event,
            contiguous=Event,
            brush_size=Event,
            selected_label=Event,
        )

        self._data_raw = np.zeros((1,) * self.dims.ndisplay)
        self._n_dimensional = False
        self._contiguous = True
        self._brush_size = 10
        self._last_cursor_coord = None

        self._selected_label = 0
        self._selected_color = None

        self._mode = Mode.PAN_ZOOM
        self._mode_history = self._mode
        self._status = self.mode
        self._help = 'enter paint or fill mode to edit labels'

        self._block_saving = False
        self._reset_history()

        # Trigger generation of view slice and thumbnail
        self._update_dims()
        self._set_editable()

        self.dims.events.ndisplay.connect(self._reset_history)
        self.dims.events.order.connect(self._reset_history)
        self.dims.events.axis.connect(self._reset_history)

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
    def brush_size(self):
        """float: Size of the paint brush."""
        return self._brush_size

    @brush_size.setter
    def brush_size(self, brush_size):
        self._brush_size = int(brush_size)
        self.cursor_size = self._brush_size / self.scale_factor
        self.status = format_float(self.brush_size)
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
        self.colormap = (
            self._colormap_name,
            colormaps.label_colormap(num_colors),
        )
        self.refresh()
        self._selected_color = self.get_color(self.selected_label)
        self.events.selected_label()

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
                'is_pyramid': self.is_pyramid,
                'num_colors': self.num_colors,
                'seed': self.seed,
                'data': self.data,
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

    @property
    def mode(self):
        """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICKER mode the cursor functions like a color picker, setting the
        clicked on label to be the curent label. If the background is picked it
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
        """
        return str(self._mode)

    @mode.setter
    def mode(self, mode: Union[str, Mode]):

        if isinstance(mode, str):
            mode = Mode(mode)

        if not self.editable:
            mode = Mode.PAN_ZOOM

        if mode == self._mode:
            return

        if mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = 'enter paint or fill mode to edit labels'
        elif mode == Mode.PICKER:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, click to pick a label'
        elif mode == Mode.PAINT:
            self.cursor_size = self.brush_size / self.scale_factor
            self.cursor = 'square'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, drag to paint a label'
        elif mode == Mode.FILL:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, click to fill a label'
        else:
            raise ValueError("Mode not recognized")

        self.status = str(mode)
        self._mode = mode

        self.events.mode(mode=mode)
        self.refresh()

    def _set_editable(self, editable=None):
        """Set editable mode based on layer properties."""
        if editable is None:
            if self.is_pyramid or self.dims.ndisplay == 3:
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
        -------
        raw : array or int
            Raw integer input image.

        Returns
        -------
        image : array
            Image mapped between 0 and 1 to be displayed.
        """
        image = np.where(
            raw > 0, colormaps._low_discrepancy_image(raw, self._seed), 0
        )
        return image

    def new_colormap(self):
        self.seed = np.random.rand()

    def get_color(self, label):
        """Return the color corresponding to a specific label."""
        if label == 0:
            col = None
        else:
            val = self._raw_to_displayed(np.array([label]))
            col = self.colormap[1][val].rgba[0]
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
            self._undo_history.append(self.data[self.dims.indices].copy())
            self._trim_history()

    def _load_history(self, before, after):
        if len(before) == 0:
            return

        prev = before.pop()
        after.append(self.data[self.dims.indices].copy())
        self.data[self.dims.indices] = prev

        self.refresh()

    def undo(self):
        self._load_history(self._undo_history, self._redo_history)

    def redo(self):
        self._load_history(self._redo_history, self._undo_history)

    def fill(self, coord, old_label, new_label):
        """Replace an existing label with a new label, either just at the
        connected component if the `contiguous` flag is `True` or everywhere
        if it is `False`, working either just in the current slice if
        the `n_dimensional` flag is `False` or on the entire data if it is
        `True`.

        Parameters
        ----------
        coord : sequence of float
            Position of mouse cursor in image coordinates.
        old_label : int
            Value of the label image at the coord to be replaced.
        new_label : int
            Value of the new label to be filled in.
        """
        self._save_history()

        int_coord = np.round(coord).astype(int)

        if self.n_dimensional or self.ndim == 2:
            # work with entire image
            labels = self.data
            slice_coord = tuple(int_coord)
        else:
            # work with just the sliced image
            labels = self._data_raw
            slice_coord = tuple(int_coord[d] for d in self.dims.displayed)

        matches = labels == old_label
        if self.contiguous:
            # if not contiguous replace only selected connected component
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
            self.data[tuple(self.dims.indices)] = labels

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

        if self.n_dimensional or self.ndim == 2:
            slice_coord = tuple(
                [
                    slice(
                        np.round(
                            np.clip(c - self.brush_size / 2 + 0.5, 0, s)
                        ).astype(int),
                        np.round(
                            np.clip(c + self.brush_size / 2 + 0.5, 0, s)
                        ).astype(int),
                        1,
                    )
                    for c, s in zip(coord, self.shape)
                ]
            )
        else:
            slice_coord = [0] * self.ndim
            for i in self.dims.displayed:
                slice_coord[i] = slice(
                    np.round(
                        np.clip(
                            coord[i] - self.brush_size / 2 + 0.5,
                            0,
                            self.shape[i],
                        )
                    ).astype(int),
                    np.round(
                        np.clip(
                            coord[i] + self.brush_size / 2 + 0.5,
                            0,
                            self.shape[i],
                        )
                    ).astype(int),
                    1,
                )
            for i in self.dims.not_displayed:
                slice_coord[i] = np.round(coord[i]).astype(int)
            slice_coord = tuple(slice_coord)

        # update the labels image
        self.data[slice_coord] = new_label

        if refresh is True:
            self.refresh()

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if self._mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self._mode == Mode.PICKER:
            self.selected_label = self._value or 0
        elif self._mode == Mode.PAINT:
            # Start painting with new label
            self._save_history()
            self._block_saving = True
            self.paint(self.coordinates, self.selected_label)
            self._last_cursor_coord = copy(self.coordinates)
        elif self._mode == Mode.FILL:
            # Fill clicked on region with new label
            self.fill(self.coordinates, self._value, self.selected_label)
        else:
            raise ValueError("Mode not recognized")

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if self._mode == Mode.PAINT and event.is_dragging:
            if self._last_cursor_coord is None:
                interp_coord = [self.coordinates]
            else:
                interp_coord = interpolate_coordinates(
                    self._last_cursor_coord, self.coordinates, self.brush_size
                )
            for c in interp_coord:
                self.paint(c, self.selected_label, refresh=False)
            self.refresh()
            self._last_cursor_coord = copy(self.coordinates)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        self._last_cursor_coord = None
        self._block_saving = False
