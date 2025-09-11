import numpy as np
from vispy.util import transforms
from vispy.visuals.transforms import MatrixTransform

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.axes import Axes
from napari.components._viewer_constants import CanvasPosition
from napari.utils.theme import get_theme
from scipy.spatial.transform import Rotation as R


class VispyZYXAxesOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        self._scale = 1

        # Target axes length in canvas pixels
        self._target_length = 80

        # We have to create the node here so we can pass it to super(),
        # otherwise a default one would be created.
        node = Axes()
        node.set_gl_state(depth_test=False, blend=True)

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
        self.viewer.camera.events.zoom.connect(self._on_zoom_change)
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
        self.node.text.text = ['Z', 'Y', 'X']

    def _on_zoom_change(self, event=None):
        """ Prevent the axes from zooming with the world. """
        scale = 1 / self.viewer.camera.zoom

        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return

        self._scale = scale
        scale = self._target_length * self._scale
        self.node.transform.scale = [scale, scale, scale, 1]

    def _on_angles_change(self, event=None):
        """Update rotation from camera angles."""
        self._on_position_change()

    def _on_position_change(self, event=None):
        """Update position and transform of the axes."""
        if self.node.parent is None:
            return

        #
        #   Translation
        #

        x_max, y_max = list(self.node.parent.size)
        position = self.overlay.position

        size = self._target_length
        x_offset = self.x_offset + size
        y_offset = self.y_offset + size
        z_offset = -1  # prevent z-fighting?

        if position == CanvasPosition.TOP_LEFT:
            translate = [x_offset, y_offset, z_offset]
        elif position == CanvasPosition.TOP_CENTER:
            translate = [x_max / 2, y_offset, z_offset]
        elif position == CanvasPosition.TOP_RIGHT:
            translate = [x_max - x_offset, y_offset, z_offset]
        elif position == CanvasPosition.BOTTOM_LEFT:
            translate = [x_offset, y_max - y_offset, z_offset]
        elif position == CanvasPosition.BOTTOM_CENTER:
            translate = [x_max / 2, y_max - y_offset, z_offset]
        elif position == CanvasPosition.BOTTOM_RIGHT:
            translate = [x_max - x_offset, y_max - y_offset, z_offset]
        else:
            # Default to bottom left
            translate = [x_offset, y_max - y_offset, z_offset]

        translation_matrix = transforms.translate(translate)

        #
        #   Rotation
        #

        rx, ry, rz = self.viewer.camera.angles
        rot = R.from_euler('zyx', [rz, ry, rx], degrees=True) # vispy uses zyx supposedly??
        view_rot_3x3 = rot.as_matrix()
        camera_orientation_3x3 = view_rot_3x3.T

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = camera_orientation_3x3

        #
        #   Scale
        #
 
        scale_matrix = transforms.scale([self._target_length, self._target_length, self._target_length])

        # order is important!
        final_transform = scale_matrix @ rotation_matrix @ translation_matrix
        self.node.transform.matrix = final_transform

    def reset(self):
        super().reset()
        self._on_data_change()
        self._on_angles_change()
