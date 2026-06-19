from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene import ViewBox
from vispy.util.quaternion import Quaternion

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.axes import Axes
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo


class _AxesScene(ViewBox):
    def __init__(self, size, font_info: FontInfo):
        self.axes = Axes(font_info=font_info)
        super().__init__(size=(size, size), bgcolor='transparent')
        self.camera = 'arcball'
        self.camera.set_state(scale_factor=2.5, fov=0)  # good baseline
        self.axes.parent = self.scene
        self.interactive = False

    def set_gl_state(self, **kwargs):
        self.axes.set_gl_state(**kwargs)


class VispyZYXAxesOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner."""

    def __init__(self, font_info: FontInfo, **kwargs) -> None:
        self._size = 80
        super().__init__(
            node=_AxesScene(size=self._size, font_info=font_info),
            **kwargs,
        )
        self.x_size = self._size
        self.y_size = self._size

        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.dashed.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_labels_text_change)
        self.overlay.events.arrows.connect(self._on_data_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(
            self._on_labels_text_change
        )
        self.viewer.camera.events.connect(self._on_angles_change)

        self.reset()

    def _on_data_change(self, event=None):
        """Update visual data like color, dashing, and arrows."""
        # Determine which axes are displayed
        axes = self.viewer.dims.displayed[::-1]

        # Counting backwards from total number of dimensions
        # determine axes positions. This is done as by default
        # the last NumPy axis corresponds to the first Vispy axis
        reversed_axes = [self.viewer.dims.ndim - 1 - a for a in axes]

        self.node.axes.set_data(
            axes=axes,
            reversed_axes=reversed_axes,
            colored=self.overlay.colored,
            bg_color=get_theme(self.viewer.theme).canvas,
            dashed=self.overlay.dashed,
            arrows=self.overlay.arrows,
        )
        self._on_labels_text_change()
        self._on_angles_change()

    def _on_labels_text_change(self):
        axes = self.viewer.dims.displayed[::-1]
        axis_labels = [self.viewer.dims.axis_labels[a] for a in axes]
        self.node.axes.text.text = axis_labels

    def _on_angles_change(self, event=None):
        """Update rotation from camera angles."""
        from scipy.spatial.transform import Rotation

        # ensure camera flip is the same as napari camera
        flip = self.viewer.camera._vispy_flipped_axes(ndisplay=3)
        self.node.camera.flip = list(flip)

        if self.viewer.dims.ndisplay == 2:
            self.node.camera.set_state(
                _quaternion=Quaternion(1, 1, 0, 0),
                center=(0.6, 0.6, 0),
                scale_factor=1.5,
            )
            return

        # NOTE: the following is just copied from VispyCamera implementation
        # flip handedness so the rotation is always righthanded even with axis flipping
        angles = self.viewer.camera.angles * np.where(flip, -1, 1)
        # undo vispy quirks (rotation of 90 digrees and lefthanded y axis)
        angles = (np.array(angles) * (1, -1, 1)) + (0, 0, 90)
        # see #8281 for why this is yzx. In short: longstanding vispy bug.
        rotation = Rotation.from_euler('yzx', angles, degrees=True)
        # Create and set quaternion
        q = Quaternion(*rotation.as_quat(scalar_first=True))

        self.node.camera.set_state(
            _quaternion=q, center=(0, 0, 0), scale_factor=2.5
        )

    def reset(self):
        super().reset()
        self._on_data_change()
