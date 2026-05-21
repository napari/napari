import numpy as np
from vispy.scene.node import Node
from vispy.visuals.transforms import MatrixTransform

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.axes import Axes
from napari.utils.theme import get_theme


class _AxesScene(Node):
    def __init__(self):
        super().__init__()
        self.axes = Axes()
        self.axes.transform = MatrixTransform()
        self.axes.parent = self

    def set_gl_state(self, **kwargs):
        self.axes.set_gl_state(**kwargs)


class VispyZYXAxesOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            node=_AxesScene(),
            **kwargs,
        )
        self._size = 80
        self.x_size = self._size
        self.y_size = self._size

        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.dashed.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_labels_change)
        self.overlay.events.arrows.connect(self._on_data_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.camera.events.angles.connect(self._on_angles_change)

        self.reset()

    def _on_data_change(self, event=None):
        """Update visual data like color, dashing, and arrows."""
        # These axes are fixed to represent ZYX, corresponding to dims 0, 1, 2.
        # The order is reversed because vispy's first axis is napari's last.
        axes = [2, 1, 0]
        reversed_axes = [0, 1, 2]

        self.node.axes.set_data(
            axes=axes,
            reversed_axes=reversed_axes,
            colored=self.overlay.colored,
            bg_color=get_theme(self.viewer.theme).canvas,
            dashed=self.overlay.dashed,
            arrows=self.overlay.arrows,
        )
        self._on_labels_change()

    def _on_labels_change(self, event=None):
        """Update text labels."""
        self.node.axes.text.visible = self.overlay.labels
        self.node.axes.text.text = ['Z', 'Y', 'X']

    def _on_angles_change(self, event=None):
        """Update rotation from camera angles."""
        from scipy.spatial.transform import Rotation as R

        rx, ry, rz = self.viewer.camera.angles
        rot = R.from_euler('xyz', [rz, ry, rx], degrees=True)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rot.as_matrix()

        self.node.axes.transform.matrix = rotation_matrix
        self.node.axes.transform.scale([self._size / 2] * 3)

    def _on_position_change(self, event=None):
        super()._on_position_change(event)
        self.node.transform.translate = self.node.transform.translate + [
            0,
            0,
            1000,
        ]

    def reset(self):
        super().reset()
        self._on_data_change()
        self._on_angles_change()
