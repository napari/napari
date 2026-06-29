from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy.scene import ViewBox
from vispy.util.quaternion import Quaternion

from napari._vispy.camera import (
    MouseToggledArcballCamera,
    get_vispy_flipped_axes,
    napari_angles_to_vispy_quat,
)
from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.axes import Axes
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.components.overlays import FloatingAxesOverlay


class _AxesScene(ViewBox):
    def __init__(self, font_info: FontInfo) -> None:
        self.axes = Axes(font_info=font_info)
        super().__init__(bgcolor='transparent', border_width=0)
        # MouseToggledPanZoomCamera fixes backspace resetting the view
        self.camera = MouseToggledArcballCamera(fov=0, interactive=False)
        self.axes.parent = self.scene
        self.interactive = False

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        self.axes.set_gl_state(*args, **kwargs)
        # TODO: fix this issue where depth is not correct but if enabled
        #       it's used to blend with the whole canvas, not just the overlay "layer"
        # self.axes.update_gl_state(depth_test=True)


class VispyFloatingAxesOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner."""

    overlay: FloatingAxesOverlay

    def __init__(self, font_info: FontInfo, **kwargs: Any) -> None:
        super().__init__(
            node=_AxesScene(font_info=font_info),
            font_info=font_info,
            **kwargs,
        )
        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.dashed.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_labels_text_change)
        self.overlay.events.arrows.connect(self._on_data_change)
        self.overlay.events.size.connect(self._on_size_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(
            self._on_labels_text_change
        )
        self.viewer.camera.events.angles.connect(self._on_angles_change)
        self.viewer.camera.events.orientation.connect(self._on_angles_change)
        self.viewer.camera.events.orientation2d.connect(self._on_angles_change)

        self.reset()

    def _on_size_change(self) -> None:
        self.node.size = (self.overlay.size, self.overlay.size)
        self.x_size = self.y_size = self.overlay.size
        # need to trigger this for re-tiling
        self._on_position_change()

    def _on_data_change(self) -> None:
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

    def _on_labels_text_change(self) -> None:
        axes = self.viewer.dims.displayed[::-1]
        axis_labels = [self.viewer.dims.axis_labels[a] for a in axes]
        self.node.axes.text.text = axis_labels
        self.node.axes.text.visible = self.overlay.labels

    def _on_angles_change(self) -> None:
        """Update rotation from camera angles."""

        # ensure camera flip is the same as napari camera
        ndisplay = self.viewer.dims.ndisplay
        flipped_axes = get_vispy_flipped_axes(
            self.viewer.camera, ndisplay=ndisplay
        )
        self.node.camera.flip = list(flipped_axes)

        if ndisplay == 2:
            # quat chosen bring the camera in line with default position.
            # -1 is because by default vispy has y going down in 2D, but
            # we're using a 3D camera for everything for simplicity.
            self.node.camera.set_state(
                _quaternion=Quaternion(1, -1, 0, 0),
                center=(0.6, 0.6, 0),
                scale_factor=1.7,  # found by testing
            )
        else:
            quat = napari_angles_to_vispy_quat(
                self.viewer.camera.angles, flipped_axes
            )
            self.node.camera.set_state(
                _quaternion=quat, center=(0, 0, 0), scale_factor=3.2
            )

    def reset(self) -> None:
        super().reset()
        self._on_size_change()
        self._on_data_change()
