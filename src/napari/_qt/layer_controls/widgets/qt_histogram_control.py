"""Collapsible histogram control for layer controls panel."""

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
from superqt import QCollapsible

from napari._qt.widgets.qt_histogram_widget import QtHistogramWidget
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Image
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers import Image


class QtHistogramControl(QtWidgetControlsBase):
    """
    Collapsible histogram control widget for Image layers.

    This widget provides a collapsible section in the layer controls
    that shows a histogram visualization along with settings controls.

    Parameters
    ----------
    parent : QWidget
        Parent widget, typically QtBaseImageControls.
    layer : Image
        The napari Image layer.

    Attributes
    ----------
    collapsible : QCollapsible
        The collapsible container widget.
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

        # Create collapsible container
        self.collapsible = QCollapsible(trans._('Histogram'), parent)
        self.collapsible.collapse()  # Start collapsed

        # Create content widget
        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # Create histogram visualization widget
        self.histogram_widget = QtHistogramWidget(layer, parent=content)
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

        content.setLayout(content_layout)
        self.collapsible.addWidget(content)

        # Connect layer events
        layer.histogram.events.log_scale.connect(self._on_log_scale_change)
        layer.histogram.events.n_bins.connect(self._on_n_bins_change)
        layer.histogram.events.mode.connect(self._on_mode_change_from_model)

        # Connect collapsible expand/collapse to enable/disable histogram
        self.collapsible.toggled.connect(self._on_collapsible_toggled)

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

    def _on_collapsible_toggled(self, expanded: bool) -> None:
        """Enable/disable histogram computation when expanded/collapsed."""
        self._layer.histogram.enabled = expanded
        if expanded:
            # Force recomputation when expanded
            self._layer.histogram.compute()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        """
        Return the collapsible widget for adding to layer controls.

        Returns
        -------
        list
            List containing a tuple of (None, collapsible widget).
            We return None for the label since the collapsible has its own title.
        """
        # Return empty label since collapsible has its own title
        return [(QtWrappedLabel(''), self.collapsible)]

    def disconnect_widget_controls(self) -> None:
        """Disconnect event handlers and clean up."""
        super().disconnect_widget_controls()
        self.histogram_widget.cleanup()
