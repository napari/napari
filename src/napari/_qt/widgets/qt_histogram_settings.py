"""Reusable histogram settings widget for log scale, mode, and bins controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QWidget,
)

from napari._qt.utils import qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr, disconnect_events
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers.image._histogram import HistogramModel


class QtHistogramSettingsWidget(QWidget):
    """Reusable widget for histogram mode and log scale.

    This widget provides the shared histogram controls used by the
    layer controls histogram panel and the contrast limits popup.

    Parameters
    ----------
    histogram_model : HistogramModel
        The histogram model to control.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    mode_combobox : QComboBox
        Combobox for selecting canvas/full mode.
    log_scale_checkbox : QCheckBox
        Checkbox for toggling log scale.
    """

    def __init__(
        self,
        histogram_model: HistogramModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._histogram = histogram_model
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Mode selector
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(['canvas', 'full'])
        self.mode_combobox.setCurrentText(histogram_model.mode)
        self.mode_combobox.setToolTip(
            trans._(
                'Compute histogram from data shown on canvas or full volume'
            )
        )
        self.mode_combobox.currentTextChanged.connect(self._on_mode_change)

        histogram_model.events.mode.connect(self._on_model_mode_change)
        layout.addWidget(self.mode_combobox)

        # Log scale checkbox
        self.log_scale_checkbox = QCheckBox(trans._('log'))
        self.log_scale_checkbox.setChecked(histogram_model.log_scale)
        self.log_scale_checkbox.setToolTip(
            trans._('Use logarithmic scale for histogram counts')
        )
        connect_setattr(
            self.log_scale_checkbox.toggled,
            histogram_model,
            'log_scale',
        )
        # Model -> UI
        histogram_model.events.log_scale.connect(
            self._on_model_log_scale_change
        )
        layout.addWidget(self.log_scale_checkbox)
        layout.addStretch()

        self.setLayout(layout)

    def _on_mode_change(self, mode: Literal['canvas', 'full']) -> None:
        """Update model when mode changes in the combobox."""
        self._histogram.mode = mode

    def _on_model_mode_change(self, event=None) -> None:
        """Update combobox when mode changes in the model."""
        with qt_signals_blocked(self.mode_combobox):
            self.mode_combobox.setCurrentText(self._histogram.mode)

    def _on_model_log_scale_change(self, event=None) -> None:
        """Update checkbox when log_scale changes in the model."""
        with qt_signals_blocked(self.log_scale_checkbox):
            self.log_scale_checkbox.setChecked(self._histogram.log_scale)

    def cleanup(self) -> None:
        """Disconnect event handlers."""
        disconnect_events(self._histogram.events, self)
