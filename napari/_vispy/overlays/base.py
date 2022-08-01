import numpy as np
from vispy.visuals.transforms import MatrixTransform, STTransform

from ...components._viewer_constants import Position
from ...utils.events import disconnect_events
from ...utils.translations import trans


class VispyBaseOverlay:
    def __init__(self, overlay, node):
        super().__init__()
        self.overlay = overlay

        self.node = node
        self.node.order = self.overlay.order

        self.overlay.events.visible.connect(self._on_visible_change)
        self.overlay.events.opacity.connect(self._on_opacity_change)

    def _on_visible_change(self):
        self.node.visible = (
            self.viewer.overlays.visible and self.overlay.visible
        )

    def _on_opacity_change(self):
        self.node.opacity = self.overlay.opacity

    def reset(self):
        self._on_visible_change()

    def close(self):
        disconnect_events(self.layer.events, self)
        self.node.transforms = MatrixTransform()
        self.node.parent = None


class VispyCanvasOverlay:
    def __init__(self, overlay, node):
        super().__init__(overlay, node)
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)

    def _on_position_change(self):
        position = self.overlay.position
        x_bar_offset, y_bar_offset = 10, 30
        canvas_size = list(self.rect_node.canvas.size)

        if position == Position.TOP_LEFT:
            sign = 1
            bar_transform = [x_bar_offset, 10, 0, 0]
        elif position == Position.TOP_RIGHT:
            sign = -1
            bar_transform = [canvas_size[0] - x_bar_offset, 10, 0, 0]
        elif position == Position.BOTTOM_RIGHT:
            sign = -1
            bar_transform = [
                canvas_size[0] - x_bar_offset,
                canvas_size[1] - y_bar_offset,
                0,
                0,
            ]
        elif position == Position.BOTTOM_LEFT:
            sign = 1
            bar_transform = [x_bar_offset, canvas_size[1] - 30, 0, 0]
        else:
            raise ValueError(
                trans._(
                    'Position {position} not recognized.',
                    deferred=True,
                    position=self.scale_bar.position,
                )
            )

        self.line_node.transform.translate = bar_transform
        scale = abs(self.line_node.transform.scale[0])
        self.line_node.transform.scale = [sign * scale, 1, 1, 1]
        self.rect_node.transform.translate = (0, 10, 0, 0)
        self.text_node.transform.translate = (0, 20, 0, 0)


class VispySceneOverlay:
    def __init__(self, overlay, node):
        super().__init__(overlay, node)
        self.node.transform = MatrixTransform()
        self.overlay.events.transform.connect(self._on_matrix_change)

    def _on_matrix_change(self):
        transform = self.layer._transforms.simplified.set_slice(
            self.layer._dims_displayed
        )
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translate = transform.translate[::-1]
        matrix = transform.linear_matrix[::-1, ::-1].T

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        affine_matrix[-1, : len(translate)] = translate

        if self._array_like and self.layer._ndisplay == 2:
            # Perform pixel offset to shift origin from top left corner
            # of pixel to center of pixel.
            # Note this offset is only required for array like data in
            # 2D.
            offset_matrix = self.layer._data_to_world.set_slice(
                self.layer._dims_displayed
            ).linear_matrix
            offset = -offset_matrix @ np.ones(offset_matrix.shape[1]) / 2
            # Convert NumPy axis ordering to VisPy axis ordering
            # and embed in full affine matrix
            affine_offset = np.eye(4)
            affine_offset[-1, : len(offset)] = offset[::-1]
            affine_matrix = affine_matrix @ affine_offset
        self._master_transform.matrix = affine_matrix
