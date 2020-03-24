from vispy.scene.visuals import Line, Compound, Mesh, Markers
from .vispy_base_layer import VispyBaseLayer
import numpy as np


class VispyShapesLayer(VispyBaseLayer):
    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Markers: corresponding to the vertices of the interaction box or the
        # shapes that are used for highlights.
        # Lines: The lines of the interaction box used for highlights.
        # Mesh: The mesh of the outlines for each shape used for highlights.
        # Mesh: The actual meshes of the shape faces and edges
        node = Compound([Mesh(), Mesh(), Line(), Markers()])

        super().__init__(layer, node)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.events.opacity.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)

        self._reset_base()
        self._on_data_change()
        self._on_highlight_change()

    def _on_data_change(self, event=None):
        faces = self.layer._data_view._mesh.displayed_triangles
        colors = self.layer._data_view._mesh.displayed_triangles_colors
        vertices = self.layer._data_view._mesh.vertices

        # Note that the indicies of the vertices need to be resversed to
        # go from numpy style to xyz
        if vertices is not None:
            vertices = vertices[:, ::-1] + 0.5

        if len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, self.layer.dims.ndisplay))
            faces = np.array([[0, 1, 2]])
            colors = np.array([[0, 0, 0, 0]])

        if self.layer.dims.ndisplay == 3 and self.layer.dims.ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        self.node._subvisuals[0].set_data(
            vertices=vertices, faces=faces, face_colors=colors
        )
        self.node.update()

    def _on_highlight_change(self, event=None):
        # Compute the vertices and faces of any shape outlines
        vertices, faces = self.layer._outline_shapes()

        if vertices is None or len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, self.layer.dims.ndisplay))
            faces = np.array([[0, 1, 2]])
        else:
            vertices = vertices + 0.5

        self.node._subvisuals[1].set_data(
            vertices=vertices, faces=faces, color=self.layer._highlight_color
        )

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            width,
        ) = self.layer._compute_vertices_and_box()

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, self.layer.dims.ndisplay))
            size = 0
        else:
            vertices = vertices + 0.5
            size = self.layer._vertex_size

        self.node._subvisuals[3].set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=1.5,
            symbol='square',
            scaling=False,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self.layer.dims.ndisplay))
            width = 0
        else:
            pos = pos + 0.5

        self.node._subvisuals[2].set_data(
            pos=pos, color=edge_color, width=width
        )

    def _on_opacity_change(self, event=None):
        pass
