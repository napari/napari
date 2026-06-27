"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.layers import Image
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components import ViewerModel


class QtHistogramContentWidget(QWidget):
    """Shared histogram visualization and settings content.

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
    viewer : ViewerModel, optional
        The napari viewer model, used for theme tracking and dock widget.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    histogram_widget : QtHistogramWidget
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Widget for log scale control.
    dock_requested : Signal
        Emitted when the user requests popping out the histogram to a dock.
    """

    dock_requested = Signal()

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

        # Settings row: log scale checkbox + dock button
        settings_row = QHBoxLayout()
        settings_row.setContentsMargins(0, 0, 0, 0)
        settings_row.setSpacing(4)

        self.settings_widget = QtHistogramSettingsWidget(
            layer.histogram,
            parent=self,
        )
        settings_row.addWidget(self.settings_widget)

        # Dock button: pop out histogram to a persistent dock widget
        self.dock_button = QPushButton('⧉')
        self.dock_button.setFixedWidth(24)
        self.dock_button.setFixedHeight(24)
        self.dock_button.setToolTip(
            trans._('Pop out histogram to a persistent dock widget')
        )
        self.dock_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.dock_button.clicked.connect(self._on_dock_clicked)
        settings_row.addWidget(self.dock_button)

        layout.addLayout(settings_row)

    def _on_dock_clicked(self) -> None:
        """Emit signal when dock button is clicked."""
        self.dock_requested.emit()

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        self.settings_widget.cleanup()
        self.histogram_widget.cleanup()
