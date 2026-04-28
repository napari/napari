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
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget

if TYPE_CHECKING:
    from napari.layers import Image


class QtHistogramControl(QtWidgetControlsBase):
    """
    Histogram control widget for Image layers.

    This widget provides the lazily-created inline histogram content that is
    shown or hidden via the histogram button on the gamma slider.

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
    settings_widget : QtHistogramSettingsWidget
        Shared widget for histogram mode and log scale controls.
    """

    def __init__(self, parent: QWidget, layer: Image) -> None:
        super().__init__(parent, layer)

        # Create content widget
        self.content_widget = QWidget(parent)
        self.content_widget.hide()
        self.histogram_content = None
        self.histogram_widget = None
        self.settings_widget = None

        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(4)
        self.content_widget.setLayout(self._content_layout)

    def ensure_content(self) -> None:
        """Create the histogram UI lazily when it is first requested."""
        if self.histogram_content is not None:
            return

        viewer = getattr(self.parent(), 'viewer', None)
        self.histogram_content = QtHistogramContentWidget(
            self._layer,
            viewer=viewer,
            parent=self.content_widget,
        )
        self._content_layout.addWidget(self.histogram_content)

        self.histogram_widget = self.histogram_content.histogram_widget
        self.settings_widget = self.histogram_content.settings_widget

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
        if self.histogram_content is not None:
            self.histogram_content.cleanup()
