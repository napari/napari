"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.layers import Image

if TYPE_CHECKING:
    from napari.components import ViewerModel


class QtHistogramContentWidget(QWidget):
    """Shared histogram visualization and settings content.

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
    viewer : ViewerModel, optional
        The napari viewer model, used for theme tracking.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    histogram_widget : QtHistogramWidget
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Widget for mode and log scale controls.
    """

    def __init__(
        self,
        layer: Image,
        viewer: ViewerModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer = layer
        self._viewer = viewer

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.histogram_widget = QtHistogramWidget(
            layer,
            viewer=viewer,
            parent=self,
        )
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
