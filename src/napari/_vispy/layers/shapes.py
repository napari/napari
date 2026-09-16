from __future__ import annotations

import typing

import numpy as np
from superqt.utils import qdebounced

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.gl import BLENDING_MODES
from napari._vispy.utils.text import update_text
from napari._vispy.visuals.shapes import ShapesVisual
from napari.settings import get_settings
from napari.utils.events import disconnect_events

if typing.TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.layers import Shapes


class VispyShapesLayer(VispyBaseLayer):
    node: ShapesVisual
    layer: Shapes

    def __init__(self, layer: Shapes, font_info: FontInfo) -> None:
        node = ShapesVisual(font_info=font_info)
        super().__init__(layer, node, font_info=font_info)

        (
            self._active_shape_vertices,
            self._active_shape_faces,
        ) = self._empty_shape_geometry()
        self._active_shape_colors = np.empty((0, 4))
        (
            self._highlight_shape_vertices,
            self._highlight_shape_faces,
        ) = self._empty_shape_geometry()

        self._on_highlight_change_debounc = qdebounced(
            self._on_highlight_change_impl
        )

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)
        self.layer.events._active_shape.connect(self._on_active_shape_change)
        self.layer.text.events.connect(self._on_text_change)
        self.layer.events.scale_factor.connect(self._update_text)

        # TODO: move to overlays
        self.node.highlight_vertices.symbol = 'square'
        self.node.highlight_vertices.scaling = False

        self.reset()
        self._on_data_change()

    def _empty_shape_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.empty((0, self.layer._slice_input.ndisplay)),
            np.empty((0, 3), dtype=int),
        )

    def _on_active_shape_change(self) -> None:
        index = self.layer._data_view.staged_index
        if index is None:
            (
                self._active_shape_vertices,
                self._active_shape_faces,
            ) = self._empty_shape_geometry()
            self._active_shape_colors = np.empty((0, 4))
            self._update_shape_highlight_mesh()
            return

        shape = self.layer._data_view.shapes[index]
        face_vertices = shape._face_vertices
        edge_vertices = (
            shape._edge_vertices + shape.edge_width * shape._edge_offsets
        )
        vertices = np.concatenate([face_vertices, edge_vertices], axis=0)
        faces = np.concatenate(
            [
                shape._face_triangles,
                shape._edge_triangles + len(face_vertices),
            ],
            axis=0,
        )
        colors = np.concatenate(
            [
                np.repeat(
                    [self.layer._data_view.face_color[index]],
                    len(shape._face_triangles),
                    axis=0,
                ),
                np.repeat(
                    [self.layer._data_view.edge_color[index]],
                    len(shape._edge_triangles),
                    axis=0,
                ),
            ],
            axis=0,
        )
        vertices = vertices[:, ::-1]
        if self.layer._slice_input.ndisplay == 3 and self.layer.ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)))

        if len(vertices) == 0 or len(faces) == 0:
            vertices, faces = self._empty_shape_geometry()
            colors = np.empty((0, 4))

        self._active_shape_vertices = vertices
        self._active_shape_faces = faces
        self._active_shape_colors = colors
        self._update_shape_highlight_mesh()

    def _update_shape_highlight_mesh(self) -> None:
        if len(self._active_shape_faces) == 0:
            if len(self._highlight_shape_faces) == 0:
                vertices = np.zeros((3, self.layer._slice_input.ndisplay))
                faces = np.array([[0, 1, 2]])
                self.node.shape_highlights.set_data(
                    vertices=vertices,
                    faces=faces,
                    face_colors=np.zeros((1, 4)),
                )
            else:
                self.node.shape_highlights.set_data(
                    vertices=self._highlight_shape_vertices,
                    faces=self._highlight_shape_faces,
                    color=self.layer._highlight_color,
                )
            self.node.update()
            return

        vertices = np.concatenate(
            [self._active_shape_vertices, self._highlight_shape_vertices],
            axis=0,
        )
        faces = np.concatenate(
            [
                self._active_shape_faces,
                self._highlight_shape_faces + len(self._active_shape_vertices),
            ],
            axis=0,
        )
        highlight_colors = np.repeat(
            [self.layer._highlight_color],
            len(self._highlight_shape_faces),
            axis=0,
        )
        colors = np.concatenate(
            [self._active_shape_colors, highlight_colors],
            axis=0,
        )

        self.node.shape_highlights.set_data(
            vertices=vertices, faces=faces, face_colors=colors
        )
        self.node.update()

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

        self.node.shape_faces.set_data(
            vertices=vertices, faces=faces, face_colors=colors
        )

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        self._update_text(update_node=False)
        self.node.update()

    def _on_highlight_change(self):
        if len(self.layer.selected_data) > 1000:
            # Defer to next frame to avoid blocking UI
            self._on_highlight_change_debounc()
        else:
            self._on_highlight_change_impl()

    def _on_highlight_change_impl(self):
        settings = get_settings()
        self.layer._highlight_width = (
            settings.appearance.highlight.highlight_thickness
        )
        self.layer._highlight_color = (
            settings.appearance.highlight.highlight_color
        )

        # Compute the vertices and faces of any shape outlines
        vertices, faces = self.layer._outline_shapes()

        if vertices is None or len(vertices) == 0 or len(faces) == 0:
            (
                self._highlight_shape_vertices,
                self._highlight_shape_faces,
            ) = self._empty_shape_geometry()
        else:
            self._highlight_shape_vertices = vertices
            self._highlight_shape_faces = faces
        self._update_shape_highlight_mesh()

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            _,
        ) = self.layer._compute_vertices_and_box()

        width = settings.appearance.highlight.highlight_thickness

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, self.layer._slice_input.ndisplay))
            size = 0
        else:
            size = self.layer._vertex_size

        self.node.highlight_vertices.set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=width,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self.layer._slice_input.ndisplay))
            width = 0

        self.node.highlight_lines.set_data(
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
        return self.node.text

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
        (
            self._active_shape_vertices,
            self._active_shape_faces,
        ) = self._empty_shape_geometry()
        self._active_shape_colors = np.empty((0, 4))
        (
            self._highlight_shape_vertices,
            self._highlight_shape_faces,
        ) = self._empty_shape_geometry()
        self._on_active_shape_change()
        self._on_highlight_change()
        self._on_blending_change()

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
