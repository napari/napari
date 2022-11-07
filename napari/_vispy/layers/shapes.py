import numpy as np

from napari._vispy.utils.gl import BLENDING_MODES
from napari._vispy.utils.text import update_text
from napari._vispy.visuals.shapes import ShapesVisual
from napari.settings import get_settings
from napari.utils.events import disconnect_events

from .base import VispyBaseLayer


class VispyShapesLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = ShapesVisual()
        super().__init__(layer, node)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)
        self.layer.text.events.connect(self._on_text_change)

        # TODO: move to overlays
        self.node._subvisuals[3].symbol = 'square'
        self.node._subvisuals[3].scaling = False

        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        faces = self.layer._data_view._mesh.displayed_triangles
        colors = self.layer._data_view._mesh.displayed_triangles_colors
        vertices = self.layer._data_view._mesh.vertices

        # Note that the indices of the vertices need to be reversed to
        # go from numpy style to xyz
        if vertices is not None:
            vertices = vertices[:, ::-1]

        if len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, self.layer._slice_input.ndisplay))
            faces = np.array([[0, 1, 2]])
            colors = np.array([[0, 0, 0, 0]])

        if (
            len(self.layer.data)
            and self.layer._slice_input.ndisplay == 3
            and self.layer.ndim == 2
        ):
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        self.node._subvisuals[0].set_data(
            vertices=vertices, faces=faces, face_colors=colors
        )

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        self._update_text(update_node=False)
        self.node.update()

    def _on_highlight_change(self):
        settings = get_settings()
        self.layer._highlight_width = settings.appearance.highlight_thickness

        # Compute the vertices and faces of any shape outlines
        vertices, faces = self.layer._outline_shapes()

        if vertices is None or len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, self.layer._slice_input.ndisplay))
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
            vertices = np.zeros((1, self.layer._slice_input.ndisplay))
            size = 0
        else:
            size = self.layer._vertex_size

        self.node._subvisuals[3].set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=width,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self.layer._slice_input.ndisplay))
            width = 0

        self.node._subvisuals[2].set_data(
            pos=pos, color=edge_color, width=width
        )

    def _update_text(self, *, update_node=True):
        """Function to update the text node properties

        Parameters
        ----------
        update_node : bool
            If true, update the node after setting the properties
        """
        update_text(node=self._get_text_node(), layer=self.layer)
        if update_node:
            self.node.update()

    def _get_text_node(self):
        """Function to get the text node from the Compound visual"""
        text_node = self.node._subvisuals[-1]
        return text_node

    def _on_text_change(self, event=None):
        if event is not None:
            if event.type == 'blending':
                self._on_blending_change(event)
                return
            if event.type == 'values':
                return
        self._update_text()

    def _on_blending_change(self):
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
