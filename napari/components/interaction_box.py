import numpy as np

from ..utils.events import EventedModel
from ..utils.events.custom_types import Array
from ._interaction_box_constants import Box


class InteractionBox(EventedModel):
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

    points: Array[float, (-1, 2)] = None
    show: bool = False
    show_handle: bool = False
    selection_box_drag: Array[float, (4, 2)] = None
    selection_box_final: Array[float, (4, 2)] = None
    angle: float = None
    show_vertices: bool = False
    _selected_vertex: int = None
    _fixed_vertex: int = None
    _fixed_aspect: float = None
    _vertex_size = 10
    _rotation_handle_length = 20
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    def __init__(self, points=None, show=False, show_handle=False):

        super().__init__(points=points, show=show, show_handle=show_handle)

    @property
    def _box(self):
        return self._create_box_from_points()

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
        if self._box is not None and self.show:
            if self.show_handle:
                box = self._box[Box.WITH_HANDLE]
            else:
                box = self._box[Box.WITHOUT_HANDLE]

            if self._selected_vertex is None:
                face_color = 'white'
            else:
                face_color = self._highlight_color

            edge_color = self._highlight_color
            vertices = box[:, ::-1]
            if self._show_vertices:
                vertices = box[:, ::-1]
            else:
                vertices = np.empty((0, 2))

            # Use a subset of the vertices of the interaction_box to plot
            # the line around the edge
            if self.show_handle:
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

    def _add_rotation_handle(self, box):
        """Adds the rotation handle to the box"""

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

        return box

    def _create_box_from_points(self):
        """Creates the axis aligned interaction box from the list of points"""
        if self.points is None or len(self.points) < 1:
            return None

        data = self.points

        min_val = np.array(
            [data[:, 0].min(axis=0), data[:, 1].min(axis=0)]
        ) - np.array([self._vertex_size / 2, self._vertex_size / 2])
        max_val = np.array(
            [data[:, 0].max(axis=0), data[:, 1].max(axis=0)]
        ) + np.array([self._vertex_size / 2, self._vertex_size / 2])
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
        return self._add_rotation_handle(box)
