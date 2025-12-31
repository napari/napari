"""Reusable histogram settings widget for log scale, mode, and bins controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from napari._qt.utils import qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components.histogram import HistogramModel


class QtHistogramSettingsWidget(QWidget):
    """Reusable widget for histogram settings (log scale, mode, bins).

    This widget provides consistent controls for histogram settings that
    can be shared between the layer controls histogram panel and the
    contrast limits popup.

    Parameters
    ----------
    histogram_model : HistogramModel
        The histogram model to control.
    parent : QWidget, optional
        Parent widget.
    show_mode : bool, default: True
        Whether to show the mode selector (displayed/full).
    show_bins : bool, default: True
        Whether to show the bins spinbox.
    show_log : bool, default: True
        Whether to show the log scale checkbox.
    compact : bool, default: False
        Use more compact layout without labels.

    Attributes
    ----------
    mode_combobox : QComboBox | None
        Combobox for selecting displayed/full mode.
    n_bins_spinbox : QSpinBox | None
        Spinbox for setting number of bins.
    log_scale_checkbox : QCheckBox | None
        Checkbox for toggling log scale.
    """

    def __init__(
        self,
        histogram_model: HistogramModel,
        parent: QWidget | None = None,
        *,
        show_mode: bool = True,
        show_bins: bool = True,
        show_log: bool = True,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self._histogram = histogram_model
        self._callbacks = []

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8 if not compact else 4)

        self.mode_combobox = None
        self.n_bins_spinbox = None
        self.log_scale_checkbox = None

        # Mode selector
        if show_mode:
            self.mode_combobox = QComboBox()
            self.mode_combobox.addItems(['displayed', 'full'])
            self.mode_combobox.setCurrentText(histogram_model.mode)
            self.mode_combobox.setToolTip(
                trans._('Compute histogram from displayed data or full volume')
            )
            self.mode_combobox.currentTextChanged.connect(self._on_mode_change)
            # Model -> UI
            histogram_model.events.mode.connect(self._on_model_mode_change)

            if not compact:
                layout.addWidget(QLabel(trans._('mode:')))
            layout.addWidget(self.mode_combobox)

        # Bins spinbox
        if show_bins:
            self.n_bins_spinbox = QSpinBox()
            self.n_bins_spinbox.setRange(8, 1024)
            self.n_bins_spinbox.setValue(histogram_model.n_bins)
            self.n_bins_spinbox.setToolTip(trans._('Number of histogram bins'))
            connect_setattr(
                self.n_bins_spinbox.valueChanged,
                histogram_model,
                'n_bins',
            )
            # Model -> UI
            histogram_model.events.n_bins.connect(self._on_model_n_bins_change)

            if not compact:
                layout.addWidget(QLabel(trans._('bins:')))
            layout.addWidget(self.n_bins_spinbox)

        # Log scale checkbox
        if show_log:
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

        self.setLayout(layout)

    def _on_mode_change(self, mode: Literal['displayed', 'full']) -> None:
        """Update model when mode changes in the combobox."""
        self._histogram.mode = mode

    def _on_model_mode_change(self, event=None) -> None:
        """Update combobox when mode changes in the model."""
        if self.mode_combobox is not None:
            with qt_signals_blocked(self.mode_combobox):
                self.mode_combobox.setCurrentText(self._histogram.mode)

    def _on_model_n_bins_change(self, event=None) -> None:
        """Update spinbox when n_bins changes in the model."""
        if self.n_bins_spinbox is not None:
            with qt_signals_blocked(self.n_bins_spinbox):
                self.n_bins_spinbox.setValue(self._histogram.n_bins)

    def _on_model_log_scale_change(self, event=None) -> None:
        """Update checkbox when log_scale changes in the model."""
        if self.log_scale_checkbox is not None:
            with qt_signals_blocked(self.log_scale_checkbox):
                self.log_scale_checkbox.setChecked(self._histogram.log_scale)

    def cleanup(self) -> None:
        """Disconnect event handlers."""
        if self.mode_combobox is not None:
            self._histogram.events.mode.disconnect(self._on_model_mode_change)
        if self.n_bins_spinbox is not None:
            self._histogram.events.n_bins.disconnect(
                self._on_model_n_bins_change
            )
        if self.log_scale_checkbox is not None:
            self._histogram.events.log_scale.disconnect(
                self._on_model_log_scale_change
            )
