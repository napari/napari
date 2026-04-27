"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.layers import Image


class QtHistogramContentWidget(QWidget):
    """Shared histogram visualization and settings content."""

    def __init__(self, layer: Image, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layer = layer

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.histogram_widget = QtHistogramWidget(layer, parent=self)
        layout.addWidget(self.histogram_widget)

        self.settings_widget = QtHistogramSettingsWidget(
            layer.histogram,
            parent=self,
        )
        layout.addWidget(self.settings_widget)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        self.settings_widget.cleanup()
        self.histogram_widget.cleanup()
