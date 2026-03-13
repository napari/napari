"""Histogram control for layer controls panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget

if TYPE_CHECKING:
    from napari.layers import Image


class QtHistogramControl(QtWidgetControlsBase):
    """
    Histogram control widget for Image layers.

    This widget provides a histogram visualization along with settings controls
    that can be shown/hidden via the histogram button on the gamma slider.

    Parameters
    ----------
    parent : QWidget
        Parent widget, typically QtBaseImageControls.
    layer : Image
        The napari Image layer.

    Attributes
    ----------
    content_widget : QWidget
        The main content widget containing histogram and controls.
    histogram_widget : QtHistogramWidget
        The vispy-based histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Shared widget for log scale, bins, and mode controls.
    """

    def __init__(self, parent: QWidget, layer: Image) -> None:
        super().__init__(parent, layer)

        # Create content widget
        self.content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # Create histogram visualization widget
        self.histogram_widget = QtHistogramWidget(
            layer, parent=self.content_widget
        )
        content_layout.addWidget(self.histogram_widget)

        # Create shared settings controls
        self.settings_widget = QtHistogramSettingsWidget(
            layer.histogram,
            parent=self.content_widget,
        )
        content_layout.addWidget(self.settings_widget)

        self.content_widget.setLayout(content_layout)

        # Start with histogram disabled (will be enabled when button is clicked)
        layer.histogram.enabled = False

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        """
        Return an empty list since this widget is dynamically added/removed.

        The histogram widget is controlled by the histogram button on the
        gamma slider and should not be added to the layer controls by default.

        Returns
        -------
        list
            Empty list - widget is not added to controls by default.
        """
        return []

    def disconnect_widget_controls(self) -> None:
        """Disconnect event handlers and clean up."""
        super().disconnect_widget_controls()
        self.settings_widget.cleanup()
        self.histogram_widget.cleanup()
