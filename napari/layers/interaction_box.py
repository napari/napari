import numpy as np

from ._constants import Box
from ..utils.event import EmitterGroup, Event


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

    def initialize_mouse_events(self, layer):
        @layer.mouse_move_callbacks.append
        def mouse_move(layer, event):
            box = self._box[Box.WITH_HANDLE]
            coord = [layer.coordinates[i] for i in layer.dims.displayed]
            distances = abs(box - coord)

            # Get the vertex sizes
            sizes = self._vertex_size * layer.scale_factor / 2

            # Check if any matching vertices
            matches = np.all(distances <= sizes, axis=1).nonzero()
            if len(matches[0]) > 0:
                self._selected_vertex = matches[0][-1]
            else:
                self._selected_vertex = None
            self.events.points_changed()
