"""Histogram control for layer controls panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_histogram_widget import QtHistogramWidget
from napari.layers import Image
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans

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
    log_scale_checkbox : QCheckBox
        Checkbox for toggling log scale.
    n_bins_spinbox : QSpinBox
        Spinbox for setting number of bins.
    mode_combobox : QComboBox
        Combobox for selecting slice vs volume mode.
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

        # Create settings controls
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(8)

        # Log scale checkbox
        self.log_scale_checkbox = QCheckBox(trans._('Log'))
        self.log_scale_checkbox.setChecked(layer.histogram.log_scale)
        self.log_scale_checkbox.setToolTip(
            trans._('Use logarithmic scale for histogram counts')
        )
        connect_setattr(
            self.log_scale_checkbox.toggled,
            layer.histogram,
            'log_scale',
        )
        settings_layout.addWidget(self.log_scale_checkbox)

        # Number of bins spinbox
        bins_label = QtWrappedLabel(trans._('Bins:'))
        self.n_bins_spinbox = QSpinBox()
        self.n_bins_spinbox.setRange(8, 1024)
        self.n_bins_spinbox.setValue(layer.histogram.n_bins)
        self.n_bins_spinbox.setToolTip(trans._('Number of histogram bins'))
        connect_setattr(
            self.n_bins_spinbox.valueChanged,
            layer.histogram,
            'n_bins',
        )
        settings_layout.addWidget(bins_label)
        settings_layout.addWidget(self.n_bins_spinbox)

        # Mode combobox
        mode_label = QtWrappedLabel(trans._('Mode:'))
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(['displayed', 'full'])
        self.mode_combobox.setCurrentText(layer.histogram.mode)
        self.mode_combobox.setToolTip(
            trans._('Compute histogram from displayed data or full volume')
        )
        self.mode_combobox.currentTextChanged.connect(self._on_mode_change)
        settings_layout.addWidget(mode_label)
        settings_layout.addWidget(self.mode_combobox)

        settings_layout.addStretch()
        content_layout.addLayout(settings_layout)

        self.content_widget.setLayout(content_layout)

        # Connect layer events
        layer.histogram.events.log_scale.connect(self._on_log_scale_change)
        layer.histogram.events.n_bins.connect(self._on_n_bins_change)
        layer.histogram.events.mode.connect(self._on_mode_change_from_model)

        # Start with histogram disabled (will be enabled when button is clicked)
        layer.histogram.enabled = False

    def _on_log_scale_change(self, event=None) -> None:
        """Update checkbox when log_scale changes in the model."""
        with qt_signals_blocked(self.log_scale_checkbox):
            self.log_scale_checkbox.setChecked(self._layer.histogram.log_scale)

    def _on_n_bins_change(self, event=None) -> None:
        """Update spinbox when n_bins changes in the model."""
        with qt_signals_blocked(self.n_bins_spinbox):
            self.n_bins_spinbox.setValue(self._layer.histogram.n_bins)

    def _on_mode_change(self, mode: str) -> None:
        """Update model when mode changes in the combobox."""
        self._layer.histogram.mode = mode

    def _on_mode_change_from_model(self, event=None) -> None:
        """Update combobox when mode changes in the model."""
        with qt_signals_blocked(self.mode_combobox):
            self.mode_combobox.setCurrentText(self._layer.histogram.mode)

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
        self.histogram_widget.cleanup()
