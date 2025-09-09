import numpy as np
from vispy.util import transforms
from vispy.visuals.transforms import MatrixTransform

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.axes import Axes
from napari.components._viewer_constants import CanvasPosition
from napari.utils.theme import get_theme


class VispyZYXAxesOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        # The user specified a fixed size, not dependent on zoom.
        # We'll use this scale value in the transform matrix.
        self.scale = 50

        # We have to create the node here so we can pass it to super(),
        # otherwise a default one would be created.
        node = Axes()

        super().__init__(
            node=node, viewer=viewer, overlay=overlay, parent=parent
        )

        # The superclass VispyCanvasOverlay gives us an STTransform, but we
        # need a MatrixTransform for 3D rotation. So, we replace it.
        self.node.transform = MatrixTransform()

        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.dashed.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_labels_change)
        self.overlay.events.arrows.connect(self._on_data_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.camera.events.angles.connect(self._on_angles_change)

        # The parent class connects _on_position_change, which we are overriding
        # to handle the MatrixTransform.

        self.reset()

    def _on_data_change(self, event=None):
        """Update visual data like color, dashing, and arrows."""
        # These axes are fixed to represent ZYX, corresponding to dims 0, 1, 2.
        # The order is reversed because vispy's first axis is napari's last.
        axes = [2, 1, 0]
        reversed_axes = [0, 1, 2]

        self.node.set_data(
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
        self.node.text.visible = self.overlay.labels
        # The labels are fixed to Z, Y, X for this overlay.
        self.node.text.text = ['Z', 'Y', 'X']

    def _on_angles_change(self, event=None):
        """Update rotation from camera angles."""
        self._on_position_change()

    def _on_position_change(self, event=None):
        """Update position and transform of the axes."""
        if self.node.parent is None:
            return

        # 1. Build the translation matrix to pin to a corner.
        x_max, y_max = list(self.node.parent.size)
        position = self.overlay.position

        # The vispy Axes visual is centered at (0,0,0) and extends from -1 to 1
        # along each axis. After scaling, it will extend from -self.scale to
        # +self.scale. We use this size to calculate the offset needed to
        # keep the visual fully in the canvas.
        size = self.scale
        x_offset = self.x_offset + size
        y_offset = self.y_offset + size

        if position == CanvasPosition.TOP_LEFT:
            translate = [x_offset, y_offset, 0]
        elif position == CanvasPosition.TOP_CENTER:
            translate = [x_max / 2, y_offset, 0]
        elif position == CanvasPosition.TOP_RIGHT:
            translate = [x_max - x_offset, y_offset, 0]
        elif position == CanvasPosition.BOTTOM_LEFT:
            translate = [x_offset, y_max - y_offset, 0]
        elif position == CanvasPosition.BOTTOM_CENTER:
            translate = [x_max / 2, y_max - y_offset, 0]
        elif position == CanvasPosition.BOTTOM_RIGHT:
            translate = [x_max - x_offset, y_max - y_offset, 0]
        else:
            # Default to bottom left
            translate = [x_offset, y_max - y_offset, 0]
        translation_matrix = transforms.translate(translate)

        # 2. Build the rotation matrix from the camera's orientation.
        # The camera's view_matrix holds the world-to-camera transformation.
        # The inverse of its rotation part aligns our axes with the camera.
        view_rot_3x3 = self.viewer.camera.view_matrix[:3, :3]
        camera_orientation_3x3 = view_rot_3x3.T

        # Embed the 3x3 rotation in a 4x4 matrix.
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = camera_orientation_3x3

        # 3. Build the scale matrix.
        scale_matrix = transforms.scale([self.scale, self.scale, self.scale])

        # 4. Combine matrices and set the node's transform.
        # The order is important: scale, then rotate, then translate.
        final_transform = translation_matrix @ rotation_matrix @ scale_matrix
        self.node.transform.matrix = final_transform
        self.node.update()

    def reset(self):
        super().reset()
        self._on_data_change()
        # Set the initial orientation and position.
        self._on_angles_change()
