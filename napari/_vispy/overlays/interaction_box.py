import numpy as np
from vispy.scene.visuals import Compound, Line, Markers

from ...components._interaction_box_constants import Box


class VispyInteractionBox:
    def __init__(self, viewer, parent=None, order=0):

        self._viewer = viewer
        self._interaction_box = viewer.overlays.interaction_box
        self.node = Compound([Line(), Markers()], parent=parent)
        self.node.order = order
        self._on_interaction_box_change()
        self._interaction_box.events.points.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.show.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.show_handle.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.show_vertices.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.transform.connect(
            self._on_interaction_box_change
        )
        self._highlight_width = 1.5

        self._vertex_size = 10
        self._rotation_handle_length = 20
        self._highlight_color = (0, 0.6, 1)

    @property
    def marker_node(self):
        """sequence of float: Scale factors."""
        return self.node._subvisuals[1]

    @property
    def line_node(self):
        """sequence of float: Scale factors."""
        return self.node._subvisuals[0]

    def _on_interaction_box_change(self, event=None):
        """Called whenever the interaction box changed."""

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            width,
        ) = self._compute_vertices_and_box()

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, self._viewer.dims.ndisplay))
            size = 0
        else:
            size = 10

        self.marker_node.set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=1.5,
            symbol='square',
            scaling=False,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self._viewer.dims.ndisplay))
            width = 0

        self.line_node.set_data(pos=pos, color=edge_color, width=width)

    def _compute_vertices_and_box(self):
        """Compute location of the box for rendering.

        Returns
        -------
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
        if (
            self._interaction_box._box is not None
            and self._interaction_box.show
        ):
            box = self._interaction_box._box
            if self._interaction_box.show_handle:
                box = self._add_rotation_handle(box)

            face_color = self._highlight_color

            edge_color = self._highlight_color
            if self._interaction_box.show_vertices:
                if self._interaction_box.show_handle:
                    vertices = box[Box.WITH_HANDLE][:, ::-1]
                else:
                    vertices = box[Box.WITHOUT_HANDLE][:, ::-1]
            else:
                vertices = np.empty((0, 2))

            # Use a subset of the vertices of the interaction_box to plot
            # the line around the edge
            if self._interaction_box.show_handle:
                pos = box[Box.LINE_HANDLE][:, ::-1]
            else:
                pos = box[Box.LINE][:, ::-1]
            width = self._highlight_width
            self._box = box
        else:
            # Otherwise show nothing
            vertices = np.empty((0, 2))
            face_color = 'white'
            edge_color = 'white'
            pos = None
            width = 0
            self._box = None

        return vertices, face_color, edge_color, pos, width

    def _add_rotation_handle(self, box):
        """Adds the rotation handle to the box"""

        if box is not None:
            rot = box[Box.TOP_CENTER]
            length_box = np.linalg.norm(
                box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT]
            )
            if length_box > 0:
                r = self._rotation_handle_length / self._viewer.camera.zoom
                rot = (
                    rot
                    - r
                    * (box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT])
                    / length_box
                )
            box = np.append(box, [rot], axis=0)

        return box
