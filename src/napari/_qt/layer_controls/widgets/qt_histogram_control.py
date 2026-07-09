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


class QtHistogramControl(QtWidgetControlsBase):  # type: ignore[metaclass]
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

        # The content_widget is a persistent container that stays in the
        # form layout at all times; it is shown/hidden via the histogram
        # button toggle, never inserted/removed from the layout.
        self.content_widget = QWidget(parent)
        self.content_widget.hide()
        # Set size policy to Ignored so the form layout doesn't reserve
        # space for the hidden widget, preventing wide controls after
        # the histogram is toggled off.
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

        # Connect to histogram model enabled event to show/hide widget
        # when the API is used directly (e.g. layer.histogram.enabled = True)
        # Use the local ``layer`` parameter (typed ``Image``) instead of
        # ``self._layer`` (typed ``Layer``) to satisfy Pylance.
        layer.histogram.events.enabled.connect(
            self._on_histogram_enabled_changed
        )

    def _on_histogram_enabled_changed(self) -> None:
        """Show/hide the histogram widget when ``enabled`` changes via the API.

        This bridges the programmatic API (``layer.histogram.enabled = True``)
        to the UI, so that enabling/disabling the histogram from code also
        shows/hides the inline widget in the layer controls.
        """
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
        """Create the histogram UI lazily when it is first requested."""
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
        """
        Return an empty list since this widget is dynamically added/removed.

        The histogram widget is controlled by the histogram button on the
        contrast limits control and should not be added to the layer controls
        by default.

        Returns
        -------
        list
            Empty list - widget is not added to controls by default.
        """
        return []

    def disconnect_widget_controls(self) -> None:
        """Disconnect event handlers and clean up."""
        disconnect_events(cast(Image, self._layer).histogram.events, self)
        super().disconnect_widget_controls()
        if self.histogram_content is not None:
            self.histogram_content.cleanup()
