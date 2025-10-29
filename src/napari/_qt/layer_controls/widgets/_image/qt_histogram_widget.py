"""Qt widget for displaying histogram using vispy visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget
from vispy.scene import SceneCanvas

from napari._vispy.visuals.histogram import HistogramVisual

if TYPE_CHECKING:
    from napari.layers import Image


class QtHistogramWidget(QWidget):
    """
    Qt widget that embeds a vispy histogram visualization.

    This widget wraps a HistogramVisual in a vispy SceneCanvas,
    providing a Qt-compatible widget that can be embedded in
    layer controls.

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    canvas : SceneCanvas
        Vispy canvas containing the histogram visual.
    view : ViewBox
        Vispy view for the histogram.
    histogram_visual : HistogramVisual
        The vispy visual that renders the histogram.
    """

    def __init__(self, layer: Image, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.layer = layer
        self._target_width = 300
        self._target_height = 150

        # Create vispy canvas
        self.canvas = SceneCanvas(
            size=(self._target_width, self._target_height),
            bgcolor='black',
            keys=None,
        )
        # Set parent after creation
        self.canvas.native.setParent(self)
        # Set reasonable height constraints
        self.canvas.native.setMinimumHeight(100)
        self.canvas.native.setMaximumHeight(150)

        # Create view - use ViewBox and add_widget pattern from working PR
        from vispy.scene import ViewBox
        self.view = ViewBox(parent=self.canvas.scene)
        self.canvas.central_widget.add_widget(self.view)

        # Create histogram visual and add to view's scene
        self.histogram_visual = HistogramVisual()
        self.histogram_visual.parent = self.view.scene

        # Set up camera
        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, 1), y=(0, 1))  # type: ignore[attr-defined]
        # Disable viewbox interaction to prevent accidental pan/zoom
        self.view.interactive = False

        # Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.canvas.native)
        self.setLayout(main_layout)

        # Connect to layer histogram events
        if hasattr(layer, 'histogram'):
            layer.histogram.events.bins.connect(self._on_histogram_change)
            layer.histogram.events.counts.connect(self._on_histogram_change)
            layer.histogram.events.log_scale.connect(self._on_histogram_change)

        # Connect to layer events that affect visualization
        if hasattr(layer, 'events'):
            layer.events.gamma.connect(self._on_gamma_change)
            layer.events.contrast_limits.connect(self._on_clims_change)

        # Initial update
        self._update_histogram()

    def _on_histogram_change(self, event=None) -> None:
        """Update visualization when histogram data changes."""
        self._update_histogram()

    def _on_gamma_change(self, event=None) -> None:
        """Update gamma curve when layer gamma changes."""
        self._update_histogram()

    def _on_clims_change(self, event=None) -> None:
        """Update contrast limit indicators when they change."""
        self._update_histogram()

    def _update_histogram(self) -> None:
        """Update the histogram visual with current data."""
        if not hasattr(self.layer, 'histogram'):
            return

        hist = self.layer.histogram
        if not hist.enabled or hist.bins is None or hist.counts is None:
            # Clear visualization if histogram is disabled or has no data
            self.histogram_visual.set_data()
            self.canvas.update()
            return

        # Get current layer properties
        gamma = getattr(self.layer, 'gamma', 1.0)
        clims = getattr(self.layer, 'contrast_limits', None)
        clims_range = getattr(self.layer, 'contrast_limits_range', None)

        # Update the visual with histogram data
        self.histogram_visual.set_data(
            bins=hist.bins,
            counts=hist.counts,
            log_scale=hist.log_scale,
            gamma=gamma,
            clims=clims,
            data_range=clims_range,
        )

        self.canvas.update()

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up resources."""
        if hasattr(self.layer, 'histogram'):
            self.layer.histogram.events.bins.disconnect(self._on_histogram_change)
            self.layer.histogram.events.counts.disconnect(self._on_histogram_change)
            self.layer.histogram.events.log_scale.disconnect(self._on_histogram_change)
        if hasattr(self.layer, 'events'):
            self.layer.events.gamma.disconnect(self._on_gamma_change)
            self.layer.events.contrast_limits.disconnect(self._on_clims_change)

        self.canvas.close()
