import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.axes import Axes
from napari.utils.theme import get_theme


class VispyAxesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    """Axes indicating world coordinate origin and orientation."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        self._scale = 1

        # Target axes length in canvas pixels
        self._target_length = 80

        super().__init__(
            node=Axes(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.dashed.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_labels_visible_change)
        self.overlay.events.arrows.connect(self._on_data_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.camera.events.zoom.connect(self._on_zoom_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(
            self._on_labels_text_change
        )

        self.reset()

    def _on_data_change(self):
        # Determine which axes are displayed
        axes = self.viewer.dims.displayed[::-1]

        # Counting backwards from total number of dimensions
        # determine axes positions. This is done as by default
        # the last NumPy axis corresponds to the first Vispy axis
        reversed_axes = [self.viewer.dims.ndim - 1 - a for a in axes]

        self.node.set_data(
            axes=axes,
            reversed_axes=reversed_axes,
            colored=self.overlay.colored,
            bg_color=get_theme(self.viewer.theme, False).canvas,
            dashed=self.overlay.dashed,
            arrows=self.overlay.arrows,
        )

        self._on_labels_text_change()

    def _on_labels_visible_change(self):
        self.node.text.visible = self.overlay.labels

    def _on_labels_text_change(self):
        axes = self.viewer.dims.displayed[::-1]
        axes_labels = [self.viewer.dims.axis_labels[a] for a in axes]
        self.node.text.text = axes_labels

    def _on_zoom_change(self):
        scale = 1 / self.viewer.camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale
        scale = self._target_length * self._scale
        # Update axes scale
        self.node.transform.reset()
        self.node.transform.scale([scale, scale, scale, 1])

    def reset(self):
        super().reset()
        self._on_data_change()
        self._on_labels_visible_change()
        self._on_zoom_change()
