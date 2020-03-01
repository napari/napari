import numpy as np
from skimage.transform import SimilarityTransform

from ._constants import Box
from ..utils.event import EmitterGroup, Event


def inside_boxes(boxes):
    """Checks which boxes contain the origin. Boxes need not be axis aligned

    Parameters
    ----------
    boxes : (N, 8, 2) array
        Array of N boxes that should be checked

    Returns
    -------
    inside : (N,) array of bool
        True if corresponding box contains the origin.
    """

    AB = boxes[:, 0] - boxes[:, 6]
    AM = boxes[:, 0]
    BC = boxes[:, 6] - boxes[:, 4]
    BM = boxes[:, 6]

    ABAM = np.multiply(AB, AM).sum(1)
    ABAB = np.multiply(AB, AB).sum(1)
    BCBM = np.multiply(BC, BM).sum(1)
    BCBC = np.multiply(BC, BC).sum(1)

    c1 = 0 <= ABAM
    c2 = ABAM <= ABAB
    c3 = 0 <= BCBM
    c4 = BCBM <= BCBC

    inside = np.all(np.array([c1, c2, c3, c4]), axis=0)

    return inside


class InteractionBox:
    """Models a box that can be used to transform an object or a set of objects

    Parameters
    ----------
    points : list
        Nx2 array of points whose interaction box is to be found
    show : bool
        Bool indicating whether the box should be drawn
    show_handle : bool
        Bool indicating whether the full box with midpoints and rotation handle should be drawn.
        If False only the corners are drawn.

    """

    _vertex_size = 10
    _rotation_handle_length = 20
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    def __init__(self, points=None, show=False, show_handle=True):
        self._box = None
        self._points = points
        self._show = show
        self._show_handle = show_handle

        self._selected_vertex = None
        self._drag_start = None
        self._drag_start_angle = None
        self._fixed_aspect = False

        self.events = EmitterGroup(
            source=self,
            auto_connect=False,
            points_changed=Event,
            rotated=Event,
            scaled=Event,
            dragged=Event,
        )
        self._create_box_from_points()
        self._add_rotation_handle()

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points
        self._create_box_from_points()
        self._add_rotation_handle()
        self.events.points_changed()

    @property
    def show(self):
        return self._show

    @show.setter
    def show(self, show):
        self._show = show
        self.events.points_changed()

    @property
    def show_handle(self):
        return self._show_handle

    @show_handle.setter
    def show_handle(self, show_handle):
        self._show_handle = show_handle
        self.events.points_changed()

    @property
    def angle(self):
        offset = self._box[Box.HANDLE] - self._box[Box.CENTER]
        angle = -np.degrees(np.arctan2(offset[0], -offset[1]))
        return angle

    def _compute_vertices_and_box(self):
        """Compute location of the box for rendering.

        Returns
        ----------
        vertices : np.ndarray
            Nx2 array of any vertices to be rendered as Markers
        face_color : str
            String of the face color of the Markers
        edge_color : str
            String of the edge color of the Markers and Line for the box
        pos : np.ndarray
            Nx2 array of vertices of the box that will be rendered using a
            Vispy Line
        width : float
            Width of the box edge
        """
        if self._box is not None and self._show:
            if self._show_handle:
                box = self._box[Box.WITH_HANDLE]
            else:
                box = self._box[Box.WITHOUT_HANDLE]

            if self._selected_vertex is None:
                face_color = 'white'
            else:
                face_color = self._highlight_color

            edge_color = self._highlight_color
            vertices = box[:, ::-1]
            # Use a subset of the vertices of the interaction_box to plot
            # the line around the edge
            if self._show_handle:
                pos = self._box[Box.LINE_HANDLE][:, ::-1]
            else:
                pos = self._box[Box.LINE][:, ::-1]
            width = self._highlight_width
        else:
            # Otherwise show nothing
            vertices = np.empty((0, 2))
            face_color = 'white'
            edge_color = 'white'
            pos = None
            width = 0

        return vertices, face_color, edge_color, pos, width

    def _add_rotation_handle(self):
        """Adds the rotation handle to the box
        """

        box = self._box

        if box is not None:
            rot = box[Box.TOP_CENTER]
            length_box = np.linalg.norm(
                box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT]
            )
            if length_box > 0:
                r = self._rotation_handle_length
                rot = (
                    rot
                    - r
                    * (box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT])
                    / length_box
                )
            box = np.append(box, [rot], axis=0)

        self._box = box

    def _create_box_from_points(self):
        """Creates the axis aligned interaction box from the list of points
        """
        if self._points is None or len(self._points) < 1:
            self._box = None
            return

        data = self._points

        min_val = [data[:, 0].min(axis=0), data[:, 1].min(axis=0)]
        max_val = [data[:, 0].max(axis=0), data[:, 1].max(axis=0)]
        tl = np.array([min_val[0], min_val[1]])
        tr = np.array([max_val[0], min_val[1]])
        br = np.array([max_val[0], max_val[1]])
        bl = np.array([min_val[0], max_val[1]])
        box = np.array(
            [
                tl,
                (tl + tr) / 2,
                tr,
                (tr + br) / 2,
                br,
                (br + bl) / 2,
                bl,
                (bl + tl) / 2,
                (tl + tr + br + bl) / 4,
            ]
        )
        self._box = box

    def _set_drag_start_values(self, layer):
        """ Gets called whenever a drag is started to remember starting values
        """

        self._drag_start_coordinates = [
            layer.coordinates[i] for i in layer.dims.displayed
        ]
        self._drag_start_box = np.copy(self._box)
        self._drag_start_angle = self.angle
        self._drag_angle = 0

    def _clear_drag_start_values(self):
        """ Gets called at the end of a drag to reset remembered values
        """

        self._drag_start_coordinates = None
        self._drag_start_box = None
        self._drag_start_angle = None
        self._drag_angle = 0

    def _drag_transform_box(self):
        """ Gets called when rotation,scale or translation change due to drag.
            The function calculates a transform that is used to transform the box
            and is passed back to the layer.
        """
        # rotate
        center = self._drag_start_box[Box.CENTER]
        tform1 = SimilarityTransform(translation=-center)
        tform2 = SimilarityTransform(rotation=-np.radians(self._drag_angle))
        tform3 = SimilarityTransform(translation=center)
        transform = tform1 + tform2 + tform3

        self._box = transform(self._drag_start_box)
        self.events.points_changed()

    def _on_rotation_drag(self, layer, event):
        """ Gets called upon mouse_move in the case of a rotation
        """

        new_offset = [
            layer.coordinates[i] for i in layer.dims.displayed
        ] - self._box[Box.CENTER]
        new_angle = -np.degrees(np.arctan2(new_offset[0], -new_offset[1]))

        if np.linalg.norm(new_offset) < 1:
            self._drag_angle = 0
        elif self._fixed_aspect:
            self._drag_angle = (
                np.round(new_angle / 45) * 45 - self._drag_start_angle
            )
        else:
            self._drag_angle = new_angle - self._drag_start_angle

        self._drag_transform_box()
        self.events.rotated(angle=self._drag_angle)

    def initialize_mouse_events(self, layer):
        """ Adds event handling functions to the layer
        """

        @layer.mouse_move_callbacks.append
        def mouse_move(layer, event):
            if not self.show or self._box is None:
                return

            box = self._box
            coord = [layer.coordinates[i] for i in layer.dims.displayed]
            distances = abs(box - coord)

            # Get the vertex sizes
            sizes = self._vertex_size * layer.scale_factor / 2

            # Check if any matching vertices
            matches = np.all(distances <= sizes, axis=1).nonzero()
            if len(matches[0]) > 0:
                self._selected_vertex = matches[0][-1]
                # Exclde center vertex
                if self._selected_vertex == Box.CENTER:
                    self._selected_vertex = None
            else:
                self._selected_vertex = None
            self.events.points_changed()

        @layer.mouse_drag_callbacks.append
        def mouse_drag(layer, event):
            if not self.show or self._box is None:
                return

            # Handling drag start, decide what action to take
            self._set_drag_start_values(layer)
            drag_callback = None
            if self._selected_vertex is not None:
                if self._selected_vertex == Box.HANDLE:
                    drag_callback = self._on_rotation_drag
                    yield
                else:
                    drag_callback = None
                    yield
            else:
                if inside_boxes(
                    np.array([self._box - self._drag_start_coordinates])
                )[0]:
                    drag_callback = None
                    yield
                else:
                    self.points = None
                    drag_callback = None
                    yield

            # Handle events during dragging
            while event.type == 'mouse_move':
                if drag_callback is not None:
                    drag_callback(layer, event)
                yield

            self._clear_drag_start_values()
