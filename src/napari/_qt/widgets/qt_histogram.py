"""Qt widget for displaying histogram using vispy visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget
from vispy.scene import SceneCanvas

from napari._vispy.visuals.histogram import HistogramVisual
from napari.settings import get_settings
from napari.utils.events.event_utils import disconnect_events
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from napari.components import ViewerModel
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
        self._histogram = layer.histogram
        self._appearance = get_settings().appearance
        self._viewer = self._find_viewer()
        self._updating = False
        self._target_width = 300
        self._target_height = 150

        theme = get_theme(self._theme_id())

        # Create vispy canvas
        self.canvas = SceneCanvas(
            size=(self._target_width, self._target_height),
            bgcolor=theme.canvas.as_hex(),
            keys=None,
        )
        # Set parent after creation
        self.canvas.native.setParent(self)
        self.canvas.native.setMinimumHeight(100)

        # Create view - use ViewBox and add_widget pattern from working PR
        from vispy.scene import ViewBox

        self.view = ViewBox(parent=self.canvas.scene)
        self.canvas.central_widget.add_widget(self.view)

        # Create histogram visual and add to view's scene
        self.histogram_visual = HistogramVisual()
        self.histogram_visual.parent = self.view.scene

        # Set up camera
        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, 1), y=(0, 1), margin=0.01)  # type: ignore[attr-defined]
        # Disable viewbox interaction to prevent accidental pan/zoom
        self.view.interactive = False

        # Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.canvas.native)
        self.setLayout(main_layout)

        # Connect to layer histogram events
        self._histogram.events.bins.connect(self._on_histogram_change)
        self._histogram.events.counts.connect(self._on_histogram_change)
        self._histogram.events.log_scale.connect(self._on_histogram_change)
        self._histogram.events.enabled.connect(self._on_histogram_change)

        # Connect to layer events that affect visualization
        layer.events.gamma.connect(self._on_gamma_change)
        layer.events.contrast_limits.connect(self._on_clims_change)
        layer.events.colormap.connect(self._on_colormap_change)

        self._appearance.events.theme.connect(self._on_theme_change)
        if self._viewer is not None:
            self._viewer.events.theme.connect(self._on_theme_change)

        self._apply_visual_style()

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

    def _on_colormap_change(self, event=None) -> None:
        """Update histogram colors when the layer colormap changes."""
        self._apply_visual_style()
        self._update_histogram()

    def _on_theme_change(self, event=None) -> None:
        """Update canvas and plot styling when the application theme changes."""
        self._apply_visual_style()
        self.canvas.update()

    def _theme_rgba(self, color, alpha: float = 1.0) -> tuple[float, ...]:
        """Convert a napari theme color to a vispy RGBA tuple."""
        red, green, blue = color.as_rgb_tuple()
        return (red / 255, green / 255, blue / 255, alpha)

    def _theme_id(self) -> str:
        """Return the active theme id for this histogram widget."""
        if self._viewer is not None:
            return self._viewer.theme
        return self._appearance.theme

    def _find_viewer(self) -> ViewerModel | None:
        """Walk parent widgets to find an owning viewer model if present."""
        parent = self.parentWidget()
        while parent is not None:
            viewer = getattr(parent, 'viewer', None)
            if viewer is not None and hasattr(viewer, 'events'):
                return viewer
            parent = parent.parentWidget()
        return None

    def _layer_bar_color(self) -> tuple[float, ...]:
        """Use the brightest end of the layer colormap for histogram bars."""
        rgba = np.atleast_2d(self.layer.colormap.map([1.0]))[0].astype(float)
        alpha = max(float(rgba[3]), 0.8)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha)

    def _apply_visual_style(self) -> None:
        """Apply theme-aware and layer-aware styling to the histogram plot."""
        theme = get_theme(self._theme_id())
        self.canvas.bgcolor = theme.canvas.as_hex()
        self.histogram_visual.set_style(
            bar_color=self._layer_bar_color(),
            lut_color=self._theme_rgba(theme.highlight, 0.95),
            axes_color=self._theme_rgba(theme.text, 0.7),
            text_color=self._theme_rgba(theme.text, 1.0),
        )

    def _update_histogram(self) -> None:
        """Update the histogram visual with current data."""
        if self._updating:
            return

        self._updating = True
        try:
            hist = self._histogram
            if not hist.enabled:
                # Clear visualization if histogram is disabled.
                self.histogram_visual.set_data()
                self.canvas.update()
                return

            bins = hist.bins
            counts = hist.counts

            # Get current layer properties
            gamma = self.layer.gamma
            clims = self.layer.contrast_limits
            clims_range = self.layer.contrast_limits_range

            # Update the visual with histogram data
            self.histogram_visual.set_data(
                bins=bins,
                counts=counts,
                log_scale=hist.log_scale,
                gamma=gamma,
                clims=clims,
                data_range=clims_range,
            )

            self.canvas.update()
        finally:
            self._updating = False

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up resources."""
        disconnect_events(self._histogram.events, self)
        disconnect_events(self.layer.events, self)
        disconnect_events(self._appearance.events, self)
        if self._viewer is not None:
            disconnect_events(self._viewer.events, self)

        self.canvas.close()
