"""Qt widget for histogram display with VisPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget
from vispy.scene import SceneCanvas, ViewBox

from napari._vispy.visuals.histogram import HistogramVisual

if TYPE_CHECKING:
    from napari.layers import Image, Surface


class QtHistogramWidget(QWidget):
    """A Qt widget containing a VisPy histogram visualization.

    This widget displays a histogram of layer data with visual indicators
    for the contrast limits. It's designed to be lightweight and performant.

    Parameters
    ----------
    layer : napari.layers.Image or napari.layers.Surface
        The layer whose data will be visualized
    parent : QWidget, optional
        Parent widget

    Attributes
    ----------
    canvas : vispy.scene.SceneCanvas
        The VisPy canvas
    view : vispy.scene.ViewBox
        The view box containing the histogram
    histogram : napari._vispy.visuals.histogram.HistogramVisual
        The histogram visual
    """

    def __init__(
        self,
        layer: Image | Surface,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer = layer

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create VisPy canvas
        self.canvas = SceneCanvas(
            size=(400, 120),
            bgcolor='black',
            keys=None,
        )
        # Set the Qt parent after creation
        self.canvas.native.setParent(self)
        # Set reasonable height constraints
        self.canvas.native.setMinimumHeight(100)
        self.canvas.native.setMaximumHeight(150)

        # Create view
        self.view = ViewBox(parent=self.canvas.scene)
        self.canvas.central_widget.add_widget(self.view)

        # Create histogram visual
        self.histogram = HistogramVisual(
            parent=self.view.scene,
            color='#666666',
            orientation='vertical',
        )

        # Set up camera - disable pan/zoom by removing viewbox interaction
        self.view.camera = 'panzoom'
        # Type ignore since camera is set as string but becomes camera object
        self.view.camera.set_range(x=(0, 1), y=(0, 1))  # type: ignore[attr-defined]
        
        # Disable viewbox interaction to prevent panning/zooming
        # but keep canvas events for our custom handlers
        self.view.interactive = False

        # Add canvas to layout
        layout.addWidget(self.canvas.native)

        # Setup mouse interaction for gamma adjustment
        self._dragging_gamma = False

        # Use canvas.connect decorator for vispy events
        @self.canvas.connect
        def on_mouse_press(event):
            """Handle mouse press to start gamma dragging."""
            if hasattr(self.layer, 'gamma') and event.button == 1:  # Left click only
                self._dragging_gamma = True

        @self.canvas.connect
        def on_mouse_move(event):
            """Handle mouse move to adjust gamma."""
            if self._dragging_gamma:
                self._on_mouse_move_impl(event)

        @self.canvas.connect
        def on_mouse_release(event):
            """Handle mouse release to stop gamma dragging."""
            if event.button == 1:  # Left click only
                self._dragging_gamma = False

        # Add log scale checkbox
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 2, 0, 0)
        self.log_checkbox = QCheckBox('Log scale')
        self.log_checkbox.setChecked(False)
        self.log_checkbox.toggled.connect(self._on_log_scale_toggled)
        controls_layout.addWidget(self.log_checkbox)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Connect to layer gamma changes if available
        if hasattr(layer, 'events') and hasattr(layer.events, 'gamma'):
            layer.events.gamma.connect(self._on_gamma_change)

        # Update histogram with layer data
        self.update_histogram()

    def _on_mouse_move_impl(self, event) -> None:
        """Handle mouse move to adjust gamma."""
        if not self._dragging_gamma or not hasattr(self.layer, 'gamma'):
            return

        # Get mouse position in data coordinates
        try:
            tr = self.view.scene.transform
            pos = tr.imap(event.pos)[:2]

            if self.histogram._clims is not None:
                clim_min, clim_max = self.histogram._clims
                max_count = self.histogram._counts.max() if self.histogram._counts is not None else 1
                
                # Account for log scale in display height
                if self.histogram.log_scale:
                    max_count = np.log10(max_count + 1)

                # Normalize mouse position
                if clim_max > clim_min and max_count > 0:
                    x_norm = (pos[0] - clim_min) / (clim_max - clim_min)
                    y_norm = pos[1] / max_count

                    # Clamp to valid range
                    x_norm = np.clip(x_norm, 0.01, 0.99)
                    y_norm = np.clip(y_norm, 0.01, 0.99)

                    # Calculate gamma from the position
                    # gamma curve: y = x^gamma, so gamma = log(y) / log(x)
                    if x_norm > 0.01 and y_norm > 0.01:
                        gamma = np.log(y_norm) / np.log(x_norm)
                        # Clamp gamma to reasonable range
                        gamma = np.clip(gamma, 0.1, 10.0)
                        self.layer.gamma = gamma  # type: ignore[attr-defined]
        except (AttributeError, ValueError, ZeroDivisionError):
            # If anything goes wrong, just ignore
            pass

    def _on_gamma_change(self) -> None:
        """Handle gamma change event from layer."""
        self.update_clim_lines()

    def _on_log_scale_toggled(self, checked: bool) -> None:
        """Handle log scale checkbox toggle."""
        self.histogram.set_log_scale(checked)
        # Update camera range for log scale
        self.update_histogram()

    def update_histogram(self) -> None:
        """Update histogram from current layer data."""
        # Get current displayed data slice
        if hasattr(self.layer, '_slice') and hasattr(self.layer, 'data'):
            try:
                # Get the current slice indices
                data = self.layer._slice.image.view  # type: ignore[attr-defined]
                if data is None or data.size == 0:
                    return

                # Compute histogram based on contrast limits range
                clim_range = tuple(self.layer.contrast_limits_range)  # type: ignore[arg-type]
                self.histogram.set_data(
                    data=data,
                    bins=128,
                    data_range=clim_range,  # type: ignore[arg-type]
                )

                # Update camera view to show full histogram with fixed bounds
                if clim_range[0] != clim_range[1] and self.histogram._counts is not None:
                    # Get the max count for Y range (accounting for log scale)
                    max_count = self.histogram._counts.max() if len(self.histogram._counts) > 0 else 1
                    if self.histogram.log_scale:
                        # Use log scale for display
                        max_count = np.log10(max_count + 1)

                    # Set camera rect to fix the view bounds
                    # rect = (x, y, width, height)
                    width = float(clim_range[1]) - float(clim_range[0])  # type: ignore[arg-type]
                    height = max_count * 1.05  # Add 5% margin at top
                    self.view.camera.rect = (clim_range[0], 0, width, height)  # type: ignore[attr-defined]
                    
                    # Ensure viewbox stays non-interactive after camera update
                    self.view.interactive = False

                # Update contrast limit lines
                self.update_clim_lines()

            except (AttributeError, IndexError, TypeError):
                # If we can't access the data, just skip
                pass

    def update_clim_lines(self) -> None:
        """Update the contrast limit indicator lines and gamma curve."""
        try:
            clims = tuple(self.layer.contrast_limits)  # type: ignore[arg-type]
            # Get gamma from layer if it has it
            gamma = getattr(self.layer, 'gamma', 1.0)
            self.histogram.set_clims(clims, gamma, self.histogram.log_scale)  # type: ignore[arg-type]
        except (AttributeError, TypeError):
            pass
