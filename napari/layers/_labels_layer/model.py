import numpy as np
from scipy import ndimage as ndi
from copy import copy
from xml.etree.ElementTree import Element
from base64 import b64encode
from imageio import imwrite

from .._base_layer import Layer
from ..._vispy.scene.visuals import Image as ImageNode

from ...util.colormaps import colormaps
from ...util.event import Event

from .._register import add_to_viewer

from .view import QtLabelsLayer
from .view import QtLabelsControls
from ._constants import Mode, BACKSPACE
from vispy.color import Colormap


@add_to_viewer
class Labels(Layer):
    """Labels (or segmentation) layer.

    An image layer where every pixel contains an integer ID corresponding
    to the region it belongs to.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel. Guesses if None.
    opacity : float, optional
        Opacity of the labels, must be between 0 and 1.
    name : str, keyword-only
        Name of the layer.
    num_colors : int, optional
        Number of unique colors to use. Default used if not given.
    **kwargs : dict
        Parameters that will be translated to metadata.
    """
    def __init__(self, label_image, meta=None, *, name=None, num_colors=50,
                 opacity=0.7, **kwargs):
        if name is None and meta is not None:
            if 'name' in meta:
                name = meta['name']

        visual = ImageNode(None, method='auto')
        super().__init__(visual, name)
        self.events.add(colormap=Event, mode=Event, n_dimensional=Event,
                        contiguous=Event, brush_size=Event,
                        selected_label=Event)

        self.seed = 0.5
        self._image = label_image
        self._image_view = None
        self._meta = meta
        self.interpolation = 'nearest'
        self.colormap_name = 'random'
        self.colormap = colormaps.label_colormap(num_colors)

        self._node.opacity = opacity
        self._n_dimensional = True
        self._contiguous = True
        self._brush_size = 10
        self._last_cursor_coord = None

        self._selected_label = 0
        self._selected_color = None

        self._mode = Mode.PAN_ZOOM
        self._mode_history = self._mode
        self._status = str(self._mode)
        self._help = 'enter paint or fill mode to edit labels'

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self._qt_properties = QtLabelsLayer(self)
        self._qt_controls = QtLabelsControls(self)

        self._node.clim = [0., 1.]
        self.events.colormap()

    def raw_to_displayed(self, raw):
        """Determines displayed image from a saved raw image and a saved seed.
        This function ensures that the 0 label gets mapped to the 0 displayed
        pixel

        Parameters
        -------
        raw : array | int
            Raw input image

        Returns
        -------
        image : array
            Image mapped between 0 and 1 to be displayed
        """
        image = np.where(raw > 0, colormaps._low_discrepancy_image(raw,
                         self.seed), 0)
        return image

    def new_colormap(self):
        self.seed = np.random.rand()

        self.refresh()

    def label_color(self, label):
        """Return the color corresponding to a specific label."""
        val = self.raw_to_displayed(np.array([label]))
        return self.colormap.map(val)

    @property
    def image(self):
        """np.ndarray: Image data.
        """
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
        self.refresh()

    @property
    def meta(self):
        """dict: Image metadata.
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta
        self.refresh()

    @property
    def data(self):
        """tuple of np.ndarray, dict: Image data and metadata.
        """
        return self.image, self.meta

    @data.setter
    def data(self, data):
        self._image, self._meta = data
        self.refresh()

    @property
    def contiguous(self):
        """ bool: if True, fill changes only pixels of the same label that are
        contiguous with the one clicked on.
        """
        return self._contiguous

    @contiguous.setter
    def contiguous(self, contiguous):
        self._contiguous = contiguous
        self.events.contiguous()

        self.refresh()

    @property
    def n_dimensional(self):
        """ bool: if True, edits labels not just in central plane but also
        in all n dimensions according to specified brush size or fill.
        """
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional):
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()

        self.refresh()

    @property
    def brush_size(self):
        """ float | list: Size of the paint brush. If a float, then if
        `n_dimensional` is False applies just to the visible dimensions, if
        `n_dimensional` is True applies to all dimensions. If a list, must be
        the same length as the number of dimensions of the layer, and size
        applies to each dimension scaled by the appropriate amount.
        """
        return self._brush_size

    @brush_size.setter
    def brush_size(self, brush_size):
        self._brush_size = int(brush_size)
        self.cursor_size = self._brush_size / self.scale_factor
        self.events.brush_size()

        self.refresh()

    @property
    def selected_label(self):
        """ int: Index of selected label. If `0` corresponds to the transparent
        background. If greater than the current maximum label then if used to
        fill or paint a region this label will be added to the new labels
        """
        return self._selected_label

    @selected_label.setter
    def selected_label(self, selected_label):
        self._selected_label = selected_label
        if selected_label == 0:
            # If background
            self._selected_color = None
        else:
            self._selected_color = self.label_color(selected_label)[0]
        self.events.selected_label()

        self.refresh()

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
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == self._mode:
            return
        old_mode = self._mode
        if mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = 'enter paint or fill mode to edit labels'
        elif mode == Mode.PICKER:
            self.cursor = 'cross'
            self.interactive = False
            self.help = ('hold <space> to pan/zoom, '
                         'click to pick a label')
        elif mode == Mode.PAINT:
            self.cursor_size = self.brush_size / self.scale_factor
            self.cursor = 'square'
            self.interactive = False
            self.help = ('hold <space> to pan/zoom, '
                         'drag to paint a label')
        elif mode == Mode.FILL:
            self.cursor = 'cross'
            self.interactive = False
            self.help = ('hold <space> to pan/zoom, '
                         'click to fill a label')
        else:
            raise ValueError("Mode not recongnized")

        self.status = str(mode)
        self._mode = mode

        self.events.mode(mode=mode)
        self.refresh()

    def _get_shape(self):
        return self.image.shape

    def _get_indices(self):
        """Gets the slice indices.

        Returns
        -------
        slice_indices : tuple
            Tuple of indices corresponding to the slice
        """
        indices = list(self.indices)

        for dim in range(len(indices)):
            max_dim_index = self.image.shape[dim] - 1

            try:
                if indices[dim] > max_dim_index:
                    indices[dim] = max_dim_index
            except TypeError:
                pass

        slice_indices = tuple(indices)
        return slice_indices

    def _slice_image(self):
        """Determines the slice of image from the indices.

        Returns
        -------
        sliced : array or value
            The requested slice.
        """
        slice_indices = self._get_indices()

        self._image_view = np.asarray(self.image[slice_indices])

        sliced = self.raw_to_displayed(self._image_view)

        return sliced

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        sliced_image = self._slice_image()
        self._node.set_data(sliced_image)

        self._need_visual_update = True
        self._update()

    @property
    def method(self):
        """string: Selects method of rendering image in case of non-linear
        transforms. Each method produces similar results, but may trade
        efficiency and accuracy. If the transform is linear, this parameter
        is ignored and a single quad is drawn around the area of the image.

            * 'auto': Automatically select 'impostor' if the image is drawn
              with a nonlinear transform; otherwise select 'subdivide'.
            * 'subdivide': ImageVisual is represented as a grid of triangles
              with texture coordinates linearly mapped.
            * 'impostor': ImageVisual is represented as a quad covering the
              entire view, with texture coordinates determined by the
              transform. This produces the best transformation results, but may
              be slow.
        """
        return self._node.method

    @method.setter
    def method(self, method):
        self._node.method = method

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
            labels = self._image
            slice_coord = tuple(int_coord)
        else:
            # work with just the sliced image
            slice_indices = self._get_indices()
            labels = self._image_view
            slice_coord = tuple(int_coord[-2:])

        matches = labels == old_label
        if self.contiguous:
            # if not contiguous replace only selected connected component
            labeled_matches, num_features = ndi.label(matches)
            if num_features != 1:
                match_label = labeled_matches[slice_coord]
                matches = np.logical_and(matches,
                                         labeled_matches == match_label)

        # Replace target pixels with new_label
        labels[matches] = new_label

        if not (self.n_dimensional or self.ndim == 2):
            # if working with just the slice, update the rest of the raw image
            self._image[slice_indices] = labels

        self.refresh()

    def _to_pix(self, pos, axis):
        """Round float from cursor position to a valid pixel

        Parameters
        ----------
        pos : float
            Float that is to be mapped.
        axis : 0 | 1
            Axis that pos corresponds to.
        Parameters
        ----------
        pix : int
            Rounded pixel value
        """

        pix = np.clip(int(round(pos)), 0, self._get_shape()[axis])
        return pix

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
            slice_coord = tuple([slice(self._to_pix(ind-self.brush_size/2, i),
                                       self._to_pix(ind+self.brush_size/2, i),
                                       1) for i, ind in enumerate(coord)])
        else:
            slice_coord = tuple(list(np.array(coord[:-2]).astype(int)) +
                                [slice(self._to_pix(ind-self.brush_size/2,
                                                    self.ndim - 2 + i),
                                       self._to_pix(ind+self.brush_size/2,
                                                    self.ndim - 2 + i), 1)
                                 for i, ind in enumerate(coord[-2:])])

        # update the labels image
        self._image[slice_coord] = new_label

        self.refresh()

    def _interp_coords(self, old_coord, new_coord):
        """Interpolates coordinates between old and new, useful for ensuring
        painting is continous. Depends on the current brush size

        Parameters
        ----------
        old_coord : np.ndarray, 1x2
            Last position of cursor.
        new_coord : np.ndarray, 1x2
            Current position of cursor.

        Returns
        ----------
        coords : np.array, Nx2
            List of coordinates to ensure painting is continous
        """
        num_step = round(max(abs(np.array(new_coord) - np.array(old_coord)))
                         / self.brush_size * 4)
        coords = [np.linspace(old_coord[i], new_coord[i],
                              num=num_step + 1) for i in range(len(new_coord))]
        coords = np.stack(coords).T
        if len(coords) > 1:
            coords = coords[1:]

        return coords

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
        coord[-2:] = np.clip(coord[-2:], 0,
                             np.asarray(self._image_view.shape) - 1)

        value = self._image_view[tuple(np.round(coord[-2:]).astype(int))]

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

    def to_xml_list(self):
        """Generates a list with a single xml element that defines the
        currently viewed image as a png according to the svg specification.

        Returns
        ----------
        xml : list of xml.etree.ElementTree.Element
            List of a single xml element specifying the currently viewed image
            as a png according to the svg specification.
        """
        image = self.raw_to_displayed(self._image_view)
        mapped_image = (self.colormap.map(image)*255).astype('uint8')
        mapped_image = mapped_image.reshape(list(self._image_view.shape) + [4])
        image_str = imwrite('<bytes>', mapped_image, format='png')
        image_str = "data:image/png;base64," + str(b64encode(image_str))[2:-1]
        props = {'xlink:href': image_str}
        width = str(self.shape[-1])
        height = str(self.shape[-2])
        opacity = str(self.opacity)
        xml = Element('image', width=width, height=height, opacity=opacity,
                      **props)
        return [xml]

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.  Converts the `event.pos`
        from canvas coordinates to `self.coordinates` in image coordinates.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.coordinates = event.pos
        coord, label = self.get_value()

        if self.mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self.mode == Mode.PICKER:
            self.selected_label = label
        elif self.mode == Mode.PAINT:
            # Start painting with new label
            new_label = self.selected_label
            self.paint(coord, new_label)
            self._last_cursor_coord = coord
            self.status = self.get_message(coord, new_label)
        elif self.mode == Mode.FILL:
            # Fill clicked on region with new label
            old_label = label
            new_label = self.selected_label
            self.fill(coord, old_label, new_label)
            self.status = self.get_message(coord, new_label)
        else:
            raise ValueError("Mode not recongnized")

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.  Converts the `event.pos`
        from canvas coordinates to `self.coordinates` in image coordinates.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.coordinates = event.pos
        coord, label = self.get_value()

        if self.mode == Mode.PAINT and event.is_dragging:
            new_label = self.selected_label
            if self._last_cursor_coord is None:
                interp_coord = [coord]
            else:
                interp_coord = self._interp_coords(self._last_cursor_coord,
                                                   coord)
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
                if self.mode != Mode.PAN_ZOOM:
                    self._mode_history = self.mode
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
