import numpy as np
from vispy.visuals.transforms import MatrixTransform, STTransform

from ...components._viewer_constants import CanvasPosition
from ...utils.events import disconnect_events
from ...utils.translations import trans


class VispyBaseOverlay:
    def __init__(self, overlay, viewer, node):
        super().__init__()
        self.overlay = overlay
        self.viewer = viewer

        self.node = node
        self.node.order = self.overlay.order

        self.overlay.events.visible.connect(self._on_visible_change)
        self.overlay.events.opacity.connect(self._on_opacity_change)

    def _on_visible_change(self):
        self.node.visible = self.overlay.visible

    def _on_opacity_change(self):
        self.node.opacity = self.overlay.opacity

    def reset(self):
        self._on_visible_change()

    def close(self):
        disconnect_events(self.layer.events, self)
        self.node.transforms = MatrixTransform()
        self.node.parent = None


class VispyCanvasOverlay(VispyBaseOverlay):
    def __init__(self, overlay, viewer, node):
        super().__init__(overlay, viewer, node)
        self.x_offset = 0
        self.y_offset = 0
        self.x_size = 0
        self.y_size = 0
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)

    def _on_position_change(self, event=None):
        if self.node.canvas is None:
            return
        x_max, y_max = list(self.node.canvas.size)
        position = self.overlay.position

        if position == CanvasPosition.TOP_LEFT:
            transform = [self.x_offset, self.y_offset, 0, 0]
        elif position == CanvasPosition.TOP_CENTER:
            transform = [
                x_max / 2 - self.x_size / 2 + self.x_offset,
                self.y_offset,
                0,
                0,
            ]
        elif position == CanvasPosition.TOP_RIGHT:
            transform = [x_max - self.x_offset, self.y_offset, 0, 0]
        elif position == CanvasPosition.BOTTOM_RIGHT:
            transform = [
                x_max - self.x_offset,
                y_max - self.y_offset,
                0,
                0,
            ]
        elif position == CanvasPosition.BOTTOM_CENTER:
            transform = [x_max // 2, y_max - self.y_offset, 0, 0]
        elif position == CanvasPosition.BOTTOM_LEFT:
            transform = [self.x_offset, y_max - self.y_offset, 0, 0]
        else:
            raise ValueError(
                trans._(
                    'Position {position} not recognized.',
                    deferred=True,
                    position=position,
                )
            )

        self.node.transform.translate = transform
        scale = abs(self.node.transform.scale[0])
        self.node.transform.scale = [scale, 1, 1, 1]


class VispySceneOverlay(VispyBaseOverlay):
    def __init__(self, overlay, viewer, node):
        super().__init__(overlay, viewer, node)
        self.node.transform = MatrixTransform()
        self.overlay.events.transform.connect(self._on_matrix_change)

    def _on_matrix_change(self, event=None):
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
