"""Qt widget for histogram display with VisPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget
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

        # Set up camera
        self.view.camera = 'panzoom'
        # Type ignore since camera is set as string but becomes camera object
        self.view.camera.set_range(x=(0, 1), y=(0, 1))  # type: ignore[attr-defined]

        # Add canvas to layout
        layout.addWidget(self.canvas.native)

        # Update histogram with layer data
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

                # Update camera view to show full histogram
                if clim_range[0] != clim_range[1] and self.histogram._counts is not None:
                    # Get the max count for Y range
                    max_count = self.histogram._counts.max() if len(self.histogram._counts) > 0 else 1
                    self.view.camera.set_range(  # type: ignore[attr-defined]
                        x=clim_range,
                        y=(0, max_count * 1.05),  # Add 5% margin at top
                        margin=0,
                    )

                # Update contrast limit lines
                self.update_clim_lines()

            except (AttributeError, IndexError, TypeError):
                # If we can't access the data, just skip
                pass

    def update_clim_lines(self) -> None:
        """Update the contrast limit indicator lines."""
        try:
            clims = tuple(self.layer.contrast_limits)  # type: ignore[arg-type]
            self.histogram.set_clims(clims)  # type: ignore[arg-type]
        except (AttributeError, TypeError):
            pass
