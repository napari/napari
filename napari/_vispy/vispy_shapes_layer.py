from vispy.scene.visuals import Line, Markers, Mesh, Compound
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
        node = Compound([Markers(), Line(), Mesh(), Mesh()])

        super().__init__(layer, node)

        self.layer.events.edge_width.connect(lambda e: self._on_data_change())
        self.layer.events.edge_color.connect(lambda e: self._on_data_change())
        self.layer.events.face_color.connect(lambda e: self._on_data_change())
        self.layer.events.opacity.connect(lambda e: self._on_data_change())
        self.layer.events.highlight.connect(
            lambda e: self._on_highlight_change()
        )

        self.reset()

    def _on_data_change(self):
        faces = self.layer._data_view._mesh.displayed_triangles
        colors = self.layer._data_view._mesh.displayed_triangles_colors
        vertices = self.layer._data_view._mesh.vertices
        if vertices is not None:
            vertices = vertices + 0.5

        if len(faces) == 0:
            self.node._subvisuals[3].set_data(vertices=None, faces=None)
        else:
            self.node._subvisuals[3].set_data(
                vertices=vertices[:, ::-1], faces=faces, face_colors=colors
            )
        self.node.update()

    def _on_highlight_change(self):
        # Compute the vertices and faces of any shape outlines
        vertices, faces = self.layer._outline_shapes()
        if vertices is not None:
            vertices = vertices + 0.5

        self.node._subvisuals[2].set_data(
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

        if vertices is not None:
            vertices = vertices + 0.5
        self.node._subvisuals[0].set_data(
            vertices,
            size=self.layer._vertex_size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=1.5,
            symbol='square',
            scaling=False,
        )

        if pos is not None:
            pos = pos + 0.5
        self.node._subvisuals[1].set_data(
            pos=pos, color=edge_color, width=width
        )

    def _on_opacity_change(self):
        pass

    def reset(self):
        self._reset_base()
        self._on_data_change()
        self._on_highlight_change()
