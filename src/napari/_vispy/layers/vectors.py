from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from vispy import use

from napari._vispy.layers.base import VispyBaseLayer

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo

logger = logging.getLogger(__name__)


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer, font_info: FontInfo) -> None:
        try:
            use(gl='gl+')  # Required for instanced rendering
            from napari._vispy.visuals.instanced_vectors import Vectors
        except RuntimeError:
            from napari._vispy.visuals.fallback_vectors import Vectors

            logger.warning(
                'Could not use gl+ for instanced rendering of vectors. '
                'Falling back to GL line rendering. This has a few limitations:\n'
                '- width is limited to integer increments\n'
                '- vectors styles other than "line" are disabled\n'
            )

        node = Vectors(font_info=font_info)
        super().__init__(layer, node, font_info=font_info)

        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.edge_width.connect(self._on_width_change)
        self.layer.events.length.connect(self._on_data_change)
        self.layer.events.vector_style.connect(self._on_data_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        # Generate simple start/end pairs for instanced rendering
        vectors_data = self.layer._view_data
        length = self.layer.length
        color = self.layer._view_face_color
        ndisplay = self.layer._slice_input.ndisplay
        ndim = self.layer.ndim

        if len(vectors_data) == 0:
            vertices = np.zeros((2, ndisplay))
        else:
            # Extract start and end points
            vectors_starts = vectors_data[:, 0]
            vectors_ends = vectors_starts + length * vectors_data[:, 1]

            # Reverse axes for vispy BEFORE interleaving
            vectors_starts = vectors_starts[:, ::-1]
            vectors_ends = vectors_ends[:, ::-1]

            # Create vertices as pairs: start0, end0, start1, end1, ...
            nvectors = len(vectors_data)
            vertices = np.zeros((2 * nvectors, vectors_data.shape[2]))
            vertices[::2] = vectors_starts
            vertices[1::2] = vectors_ends

        if ndisplay == 3 and ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        # Set data first (which includes rebinding all buffers)
        # Pass face_colors directly - the visual handles per-instance color indexing
        self.node.set_data(
            vertices=vertices,
            colors=color,
            vector_style=self.layer.vector_style,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_width_change(self):
        self.node.width = self.layer.edge_width
