import numpy as np

from ...settings import get_settings
from ...utils.events import disconnect_events
from ..utils.gl import BLENDING_MODES
from ..utils.text import update_text
from ..visuals.shapes import ShapesVisual
from .base import VispyBaseLayer


class VispyShapesLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = ShapesVisual()
        super().__init__(layer, node)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.text._connect_update_events(
            self._on_text_change, self._on_blending_change
        )
        self.layer.events.highlight.connect(self._on_highlight_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self, event=None):
        faces = self.layer._data_view._mesh.displayed_triangles
        colors = self.layer._data_view._mesh.displayed_triangles_colors
        vertices = self.layer._data_view._mesh.vertices

        # Note that the indices of the vertices need to be reversed to
        # go from numpy style to xyz
        if vertices is not None:
            vertices = vertices[:, ::-1]

        if len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, self.layer._ndisplay))
            faces = np.array([[0, 1, 2]])
            colors = np.array([[0, 0, 0, 0]])

        if self.layer._ndisplay == 3 and self.layer.ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        self.node._subvisuals[0].set_data(
            vertices=vertices, faces=faces, face_colors=colors
        )

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        self._on_text_change(update_node=False)
        self.node.update()

    def _on_highlight_change(self, event=None):
        settings = get_settings()
        self.layer._highlight_width = settings.appearance.highlight_thickness

        # Compute the vertices and faces of any shape outlines
        vertices, faces = self.layer._outline_shapes()

        if vertices is None or len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, self.layer._ndisplay))
            faces = np.array([[0, 1, 2]])

        self.node._subvisuals[1].set_data(
            vertices=vertices,
            faces=faces,
            color=self.layer._highlight_color,
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

        width = settings.appearance.highlight_thickness

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, self.layer._ndisplay))
            size = 0
        else:
            size = self.layer._vertex_size

        self.node._subvisuals[3].set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=width,
            symbol='square',
            scaling=False,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self.layer._ndisplay))
            width = 0

        self.node._subvisuals[2].set_data(
            pos=pos, color=edge_color, width=width
        )

    def _on_text_change(self, update_node=True):
        """Function to update the text node properties

        Parameters
        ----------
        update_node : bool
            If true, update the node after setting the properties
        """
        ndisplay = self.layer._ndisplay
        if (len(self.layer._indices_view) == 0) or (
            self.layer._text.visible is False
        ):
            text_coords = np.zeros((1, ndisplay))
            text = []
            anchor_x = 'center'
            anchor_y = 'center'
        else:
            text_coords, anchor_x, anchor_y = self.layer._view_text_coords
            if len(text_coords) == 0:
                text_coords = np.zeros((1, ndisplay))
            text = self.layer._view_text
        text_node = self._get_text_node()
        update_text(
            text_values=text,
            coords=text_coords,
            anchor=(anchor_x, anchor_y),
            rotation=self.layer._text.rotation,
            color=self.layer._text.color,
            size=self.layer._text.size,
            ndisplay=ndisplay,
            text_node=text_node,
        )
        if update_node:
            self.node.update()

    def _get_text_node(self):
        """Function to get the text node from the Compound visual"""
        text_node = self.node._subvisuals[-1]
        return text_node

    def _on_blending_change(self, event=None):
        """Function to set the blending mode"""
        shapes_blending_kwargs = BLENDING_MODES[self.layer.blending]
        self.node.set_gl_state(**shapes_blending_kwargs)

        text_node = self._get_text_node()
        text_blending_kwargs = BLENDING_MODES[self.layer.text.blending]
        text_node.set_gl_state(**text_blending_kwargs)
        self.node.update()

    def reset(self):
        super().reset()
        self._on_highlight_change()
        self._on_blending_change()

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
