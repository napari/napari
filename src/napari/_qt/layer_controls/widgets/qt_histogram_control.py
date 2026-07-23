"""Histogram control for layer controls panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qtpy.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari.layers import Image
from napari.utils.events import disconnect_events

if TYPE_CHECKING:
    from napari._qt.widgets.qt_histogram import QtHistogramWidget
    from napari._qt.widgets.qt_histogram_settings import (
        QtHistogramSettingsWidget,
    )


class QtHistogramControl(QtWidgetControlsBase):
    """
    Histogram control widget for Image layers.

    This widget provides the lazily-created inline histogram content that is
    shown or hidden via the histogram button on the contrast limits control.

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
    histogram_content : QtHistogramContentWidget or None
        The lazy-created histogram content instance.
    histogram_widget : QtHistogramWidget or None
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget or None
        Widget for histogram mode and log scale controls.
    """

    def __init__(self, parent: QWidget, layer: Image) -> None:
        super().__init__(parent, layer)

        # Persistent container — always in the form layout, shown/hidden
        # via button toggle, never inserted/removed at runtime.
        self.content_widget = QWidget(parent)
        self.content_widget.hide()
        # Ignored size policy prevents layout from reserving space when hidden.
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.histogram_content: QtHistogramContentWidget | None = None
        self.histogram_widget: QtHistogramWidget | None = None
        self.settings_widget: QtHistogramSettingsWidget | None = None

        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(4)
        self.content_widget.setLayout(self._content_layout)

        # Bridge API-enabled changes to UI visibility.
        layer.histogram.events.enabled.connect(
            self._on_histogram_enabled_changed
        )

    def _on_histogram_enabled_changed(self) -> None:
        """Show/hide histogram when enabled changes via the API."""
        layer = cast(Image, self._layer)
        if layer.histogram.enabled:
            self.ensure_content()
            self.content_widget.show()
            self.content_widget.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred,
            )
        else:
            self.content_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Ignored,
            )
            self.content_widget.hide()

    def ensure_content(self) -> None:
        """Lazy-create histogram UI on first request."""
        if self.histogram_content is not None:
            return

        self.histogram_content = QtHistogramContentWidget(
            cast(Image, self._layer),
            parent=self.content_widget,
        )
        self._content_layout.addWidget(self.histogram_content)

        self.histogram_widget = self.histogram_content.histogram_widget
        self.settings_widget = self.histogram_content.settings_widget

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        """Return empty list; histogram is dynamically shown/hidden."""
        return []

    def disconnect_widget_controls(self) -> None:
        """Disconnect event handlers and clean up."""
        disconnect_events(cast(Image, self._layer).histogram.events, self)
        super().disconnect_widget_controls()
        if self.histogram_content is not None:
            self.histogram_content.cleanup()
