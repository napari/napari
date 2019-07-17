from typing import Union
import warnings
import numpy as np
from scipy import ndimage as ndi
from xml.etree.ElementTree import Element
from base64 import b64encode
from imageio import imwrite

from ..base import Layer
from vispy.scene.visuals import Image as ImageNode
from ...util.colormaps import colormaps
from ...util.event import Event
from ...util.misc import interpolate_coordinates
from ._constants import Mode


class Labels(Layer):
    """Labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    labels : array
        Labels data.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap.
    seed : float
        Seed for colormap random generator.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    name : str
        Name of the layer.

    Attributes
    ----------
    data : array
        Integer valued label data. Can be N dimensional. Every pixel contains
        an integer ID corresponding to the region it belongs to. The label 0 is
        rendered as transparent.
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
    _data_view : array (N, M)
        2D labels data for the currently viewed slice.
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    _last_cursor_coord : list or None
        Coordinates of last cursor click before painting, gets reset to None
        after painting is done. Used for interpolating brush strokes.
    """

    class_keymap = {}

    def __init__(
        self,
        labels,
        *,
        metadata=None,
        num_colors=50,
        seed=0.5,
        opacity=0.7,
        name=None,
        **kwargs,
    ):

        visual = ImageNode(None, method='auto')
        super().__init__(visual, name)
        self.events.add(
            colormap=Event,
            mode=Event,
            n_dimensional=Event,
            contiguous=Event,
            brush_size=Event,
            selected_label=Event,
        )

        self._data = labels
        self._data_view = np.zeros((1, 1))
        self.metadata = metadata or {}
        self._seed = seed

        self._colormap_name = 'random'
        self._num_colors = num_colors
        self.colormap = (
            self._colormap_name,
            colormaps.label_colormap(self.num_colors),
        )
        self._node.clim = [0.0, 1.0]
        self._node._cmap = self.colormap[1]

        self._node.opacity = opacity
        self._n_dimensional = True
        self._contiguous = True
        self._brush_size = 10
        self._last_cursor_coord = None

        self._selected_label = 0
        self._selected_color = None

        self._mode = Mode.PAN_ZOOM
        self._mode_history = self._mode
        self._status = self.mode
        self._help = 'enter paint or fill mode to edit labels'

        # update flags
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
        """array: Labels data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.events.data()
        self.refresh()

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
        self.events.brush_size()

    @property
    def seed(self):
        """float: Seed for colormap random generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.refresh()

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

    @property
    def selected_label(self):
        """int: Index of selected label."""
        return self._selected_label

    @selected_label.setter
    def selected_label(self, selected_label):
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

        if mode == self._mode:
            return

        if mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = 'enter paint or fill mode to edit labels'
        elif mode == Mode.PICKER:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, ' 'click to pick a label'
        elif mode == Mode.PAINT:
            self.cursor_size = self.brush_size / self.scale_factor
            self.cursor = 'square'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, ' 'drag to paint a label'
        elif mode == Mode.FILL:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, ' 'click to fill a label'
        else:
            raise ValueError("Mode not recongnized")

        self.status = str(mode)
        self._mode = mode

        self.events.mode(mode=mode)
        self.refresh()

    def _get_shape(self):
        return self.data.shape

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
        self._seed = np.random.rand()
        self._selected_color = self.get_color(self.selected_label)
        self.refresh()

    def get_color(self, label):
        """Return the color corresponding to a specific label."""
        if label == 0:
            col = None
        else:
            val = self._raw_to_displayed(np.array([label]))
            col = self.colormap[1].map(val)[0]
        return col

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        indices = list(self.indices)
        indices[:-2] = np.clip(
            indices[:-2], 0, np.subtract(self.shape[:-2], 1)
        )
        self._data_view = np.asarray(self.data[tuple(indices)])

        image = self._raw_to_displayed(self._data_view)
        self._node.set_data(image)

        self._need_visual_update = True
        self._update()

        coord, label = self.get_value()
        self.status = self.get_message(coord, label)
        self._update_thumbnail()

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
        int_coord = np.round(coord).astype(int)

        if self.n_dimensional or self.ndim == 2:
            # work with entire image
            labels = self.data
            slice_coord = tuple(int_coord)
        else:
            # work with just the sliced image
            labels = self._data_view
            slice_coord = tuple(int_coord[-2:])

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
            indices = list(self.indices)
            indices[:-2] = np.clip(
                indices[:-2], 0, np.subtract(self.shape[:-2], 1)
            )
            self.data[tuple(indices)] = labels

        self.refresh()

    def paint(self, coord, new_label):
        """Paint over existing labels with a new label, using the selected
        brush shape and size, either only on the visible slice or in all
        n dimensions.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        """
        if self.n_dimensional or self.ndim == 2:
            slice_coord = tuple(
                [
                    slice(
                        np.round(
                            np.clip(c - self.brush_size / 2, 0, s)
                        ).astype(int),
                        np.round(
                            np.clip(c + self.brush_size / 2, 0, s)
                        ).astype(int),
                        1,
                    )
                    for c, s in zip(coord, self.shape)
                ]
            )
        else:
            slice_coord = tuple(
                list(np.array(coord[:-2]).astype(int))
                + [
                    slice(
                        np.round(
                            np.clip(c - self.brush_size / 2, 0, s)
                        ).astype(int),
                        np.round(
                            np.clip(c + self.brush_size / 2, 0, s)
                        ).astype(int),
                        1,
                    )
                    for c, s in zip(coord[-2:], self.shape[-2:])
                ]
            )

        # update the labels image
        self.data[slice_coord] = new_label

        self.refresh()

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        coord : sequence of float
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.
        """
        coord = list(self.coordinates)
        coord[-2:] = np.clip(
            coord[-2:], 0, np.asarray(self._data_view.shape) - 1
        )

        value = self._data_view[tuple(np.round(coord[-2:]).astype(int))]

        return coord, value

    def get_message(self, coord, value):
        """Generates a string based on the coordinates and information about
        what shapes are hovered over

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        value : int
            Value of the label image at the coord.

        Returns
        ----------
        msg : string
            String containing a message that can be used as a status update.
        """
        int_coord = np.round(coord).astype(int)
        msg = f'{int_coord}, {self.name}, label {value}'

        return msg

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colors.
        """
        zoom_factor = np.divide(
            self._thumbnail_shape[:2], self._data_view.shape[:2]
        ).min()
        # warning filter can be removed with scipy 1.4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downsampled = np.round(
                ndi.zoom(
                    self._data_view, zoom_factor, prefilter=False, order=0
                )
            )
        downsampled = self._raw_to_displayed(downsampled)
        colormapped = self.colormap[1].map(downsampled)
        colormapped = colormapped.reshape(downsampled.shape + (4,))
        # render background as black instead of transparent
        colormapped[..., 3] = 1
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def to_xml_list(self):
        """Generates a list with a single xml element that defines the
        currently viewed image as a png according to the svg specification.

        Returns
        ----------
        xml : list of xml.etree.ElementTree.Element
            List of a single xml element specifying the currently viewed image
            as a png according to the svg specification.
        """
        image = self._raw_to_displayed(self._data_view)
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

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord, label = self.get_value()

        if self._mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self._mode == Mode.PICKER:
            self.selected_label = label
        elif self._mode == Mode.PAINT:
            # Start painting with new label
            new_label = self.selected_label
            self.paint(coord, new_label)
            self._last_cursor_coord = coord
            self.status = self.get_message(coord, new_label)
        elif self._mode == Mode.FILL:
            # Fill clicked on region with new label
            old_label = label
            new_label = self.selected_label
            self.fill(coord, old_label, new_label)
            self.status = self.get_message(coord, new_label)
        else:
            raise ValueError("Mode not recongnized")

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord, label = self.get_value()

        if self._mode == Mode.PAINT and event.is_dragging:
            new_label = self.selected_label
            if self._last_cursor_coord is None:
                interp_coord = [coord]
            else:
                interp_coord = interpolate_coordinates(
                    self._last_cursor_coord, coord, self.brush_size
                )
            with self.freeze_refresh():
                for c in interp_coord:
                    self.paint(c, new_label)
            self.refresh()
            self._last_cursor_coord = coord
            label = new_label

        self.status = self.get_message(coord, label)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        self._last_cursor_coord = None

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self._mode != Mode.PAN_ZOOM:
                    self._mode_history = self._mode
                    self.mode = Mode.PAN_ZOOM
                else:
                    self._mode_history = Mode.PAN_ZOOM
            elif event.key == 'p':
                self.mode = Mode.PAINT
            elif event.key == 'f':
                self.mode = Mode.FILL
            elif event.key == 'z':
                self.mode = Mode.PAN_ZOOM
            elif event.key == 'l':
                self.mode = Mode.PICKER
            elif event.key == 'e':
                self.selected_label = 0
            elif event.key == 'm':
                self.selected_label = self.data.max() + 1
            elif event.key == 'd':
                if self.selected_label > 0:
                    self.selected_label = self.selected_label - 1
            elif event.key == 'i':
                self.selected_label = self.selected_label + 1

    def on_key_release(self, event):
        """Called whenever key released in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.key == ' ':
            if self._mode_history != Mode.PAN_ZOOM:
                self.mode = self._mode_history
