import contextlib
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)
from superqt import QDoubleRangeSlider

from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.layers import Image, Surface
from napari.utils._dtype import normalize_dtype
from napari.utils.events.event_utils import connect_no_arg, connect_setattr
from napari.utils.translations import trans


def range_to_decimals(range_, dtype):
    """Convert a range to decimals of precision.

    Parameters
    ----------
    range_ : tuple
        Slider range, min and then max values.
    dtype : np.dtype
        Data type of the layer. Integers layers are given integer.
        step sizes.

    Returns
    -------
    int
        Decimals of precision.
    """
    dtype = normalize_dtype(dtype)

    if np.issubdtype(dtype, np.integer):
        return 0

    # scale precision with the log of the data range order of magnitude
    # eg.   0 - 1   (0 order of mag)  -> 3 decimal places
    #       0 - 10  (1 order of mag)  -> 2 decimals
    #       0 - 100 (2 orders of mag) -> 1 decimal
    #       ≥ 3 orders of mag -> no decimals
    # no more than 64 decimals
    d_range = np.subtract(*range_[::-1])
    return min(64, max(int(3 - np.log10(d_range)), 0))


class _QDoubleRangeSlider(QDoubleRangeSlider):
    show_clim_popup = Signal()

    def mousePressEvent(self, event):
        """Update the slider, or, on right-click, pop-up an expanded slider.

        The expanded slider provides finer control, directly editable values,
        and the ability to change the available range of the sliders.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.button() == Qt.MouseButton.RightButton:
            self.show_clim_popup.emit()
        else:
            super().mousePressEvent(event)


class QContrastLimitsPopup(QtPopup):
    """Popup for contrast limits with histogram visualization.

    Unlike the simple QRangeSliderPopup, this uses a vertical layout
    to stack the slider, histogram, and controls vertically.
    """

    def __init__(
        self, layer: Image | Surface, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)

        self._layer = layer

        # Create vertical layout for stacking widgets
        from qtpy.QtWidgets import QApplication, QVBoxLayout
        from superqt import QLabeledDoubleRangeSlider

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(5)
        self.frame.setLayout(self._layout)

        # Create slider
        self.slider = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal, parent
        )
        self.slider.label_shift_x = 2
        self.slider.label_shift_y = 2
        self.slider.setFocus()

        decimals = range_to_decimals(layer.contrast_limits_range, layer.dtype)
        self.slider.setRange(*layer.contrast_limits_range)
        self.slider.setDecimals(decimals)
        self.slider.setSingleStep(10**-decimals)
        self.slider.setValue(layer.contrast_limits)

        # Add slider to layout
        self._layout.addWidget(self.slider)
        QApplication.processEvents()
        self.slider._reposition_labels()

        connect_setattr(self.slider.valueChanged, layer, 'contrast_limits')
        connect_setattr(
            self.slider.rangeChanged, layer, 'contrast_limits_range'
        )

        # Add histogram widget for Image layers, Surface layers not yet implemented
        self.histogram_widget = None
        self.settings_widget = None
        if isinstance(layer, Image) and hasattr(layer, 'histogram'):
            # Create histogram widget
            self.histogram_widget = QtHistogramWidget(layer)

            # Add histogram below the slider (slider is at index 0)
            self._layout.insertWidget(1, self.histogram_widget)

            # Create controls layout below histogram
            controls_layout = QHBoxLayout()
            controls_layout.setContentsMargins(0, 5, 0, 5)
            controls_layout.setSpacing(10)

            # Add shared settings widget (mode, bins, log scale)
            self.settings_widget = QtHistogramSettingsWidget(
                layer.histogram,
                show_mode=True,
                show_bins=True,
                show_log=True,
                compact=True,
            )
            controls_layout.addWidget(self.settings_widget)

            # Add gamma slider with label
            if hasattr(layer, 'gamma'):
                gamma_label = QLabel('gamma:')
                controls_layout.addWidget(gamma_label)

                self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
                self.gamma_slider.setMinimum(1)  # 0.1
                self.gamma_slider.setMaximum(20)  # 2.0
                self.gamma_slider.setValue(int(layer.gamma * 10))
                self.gamma_slider.setFixedWidth(150)
                self.gamma_slider.setToolTip(
                    trans._('Adjust gamma correction (0.1 - 2.0)')
                )
                self.gamma_slider.valueChanged.connect(
                    self._on_gamma_slider_changed
                )
                controls_layout.addWidget(self.gamma_slider)

                self.gamma_value_label = QLabel(f'{layer.gamma:.2f}')
                self.gamma_value_label.setFixedWidth(35)
                controls_layout.addWidget(self.gamma_value_label)

                # Connect layer gamma changes to update slider
                layer.events.gamma.connect(self._on_layer_gamma_changed)

            controls_layout.addStretch()

            # Add controls layout to main layout below histogram
            self._layout.insertWidget(
                2, self._create_widget_from_layout(controls_layout)
            )

        # Create button row for reset and full range buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 5, 0, 0)
        button_layout.setSpacing(5)

        def reset():
            layer.reset_contrast_limits()
            layer.contrast_limits_range = layer.contrast_limits
            decimals_ = range_to_decimals(
                layer.contrast_limits_range, layer.dtype
            )
            self.slider.setDecimals(decimals_)
            self.slider.setSingleStep(10**-decimals_)
            # Update histogram after reset
            if self.histogram_widget is not None:
                layer.histogram.compute()

        reset_btn = QPushButton('reset')
        reset_btn.setObjectName('reset_clims_button')
        reset_btn.setToolTip(trans._('Autoscale contrast to data range'))
        reset_btn.setFixedWidth(45)
        reset_btn.clicked.connect(reset)
        button_layout.addWidget(reset_btn)

        # the "full range" button doesn't do anything if it's not an
        # unsigned integer type (it's unclear what range should be set)
        # so we don't show create it at all.
        if np.issubdtype(normalize_dtype(layer.dtype), np.integer):
            range_btn = QPushButton('full range')
            range_btn.setObjectName('full_clim_range_button')
            range_btn.setToolTip(
                trans._('Set contrast range to full bit-depth')
            )
            range_btn.setFixedWidth(75)
            range_btn.clicked.connect(layer.reset_contrast_limits_range)
            button_layout.addWidget(range_btn)

        button_layout.addStretch()

        # Add button row to main layout
        self._layout.addWidget(self._create_widget_from_layout(button_layout))

    def keyPressEvent(self, event):
        """On key press lose focus of the lineEdits."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.slider.setFocus()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up event handlers when popup is closed."""
        self._cleanup()
        super().closeEvent(event)

    def hideEvent(self, event):
        """Clean up event handlers when popup is hidden."""
        self._cleanup()
        super().hideEvent(event)

    def _cleanup(self) -> None:
        """Disconnect event handlers and clean up widgets."""
        if self.settings_widget is not None:
            self.settings_widget.cleanup()
        if self.histogram_widget is not None:
            self.histogram_widget.cleanup()
        if hasattr(self, 'gamma_slider') and hasattr(self._layer, 'events'):
            with contextlib.suppress(ValueError, RuntimeError):
                self._layer.events.gamma.disconnect(
                    self._on_layer_gamma_changed
                )

    def _create_widget_from_layout(self, layout: QHBoxLayout) -> QWidget:
        """Helper to wrap a layout in a widget."""
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _on_gamma_slider_changed(self, value: int) -> None:
        """Handle gamma slider value change."""
        gamma = (
            value / 10.0
        )  # Convert slider value (10-100) to gamma (0.1-10.0)
        if self.histogram_widget is not None and hasattr(
            self.histogram_widget.layer, 'gamma'
        ):
            self.histogram_widget.layer.gamma = gamma
            self.gamma_value_label.setText(f'{gamma:.2f}')

    def _on_layer_gamma_changed(self, event=None) -> None:
        """Update gamma slider when layer gamma changes externally."""
        if hasattr(self, 'gamma_slider') and self.histogram_widget is not None:
            layer = self.histogram_widget.layer
            if hasattr(layer, 'gamma'):
                with qt_signals_blocked(self.gamma_slider):
                    self.gamma_slider.setValue(int(layer.gamma * 10))
                self.gamma_value_label.setText(f'{layer.gamma:.2f}')


class AutoScaleButtons(QWidget):
    def __init__(
        self, layer: Image | Surface, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent=parent)

        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)
        once_btn = QPushButton(trans._('once'))
        once_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        auto_btn = QPushButton(trans._('continuous'))
        auto_btn.setCheckable(True)
        auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        once_btn.clicked.connect(lambda: auto_btn.setChecked(False))
        connect_no_arg(once_btn.clicked, layer, 'reset_contrast_limits')
        connect_setattr(auto_btn.toggled, layer, '_keep_auto_contrast')
        connect_no_arg(auto_btn.clicked, layer, 'reset_contrast_limits')

        self.layout().addWidget(once_btn)
        self.layout().addWidget(auto_btn)

        # just for testing
        self._once_btn = once_btn
        self._auto_btn = auto_btn


class QtContrastLimitsControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer contrast
    limits/autocontrast and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Image | napari.layers.Surface
        An instance of a napari layer.

    Attributes
    ----------
    auto_scale_bar : AutoScaleButtons
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    auto_scale_bar_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the auto-contrast functionality widget.
    clim_popup : napari._qt.qt_range_slider_popup.QRangeSliderPopup
        Popup widget launching the contrast range slider.
    contrast_limits_slider : _QDoubleRangeSlider
        Slider controlling current constrast limits of the layer.
    contrast_limits_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the constrast limits chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Image | Surface) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self._layer.events.contrast_limits_range.connect(
            self._on_contrast_limits_range_change
        )

        # Setup widgets
        self.auto_scale_bar = AutoScaleButtons(layer, parent)
        self.auto_scale_bar_label = QtWrappedLabel(trans._('auto-contrast:'))
        self.contrast_limits_slider = _QDoubleRangeSlider(
            Qt.Orientation.Horizontal,
        )
        self.contrast_limits_slider.show_clim_popup.connect(
            self.show_clim_popup
        )
        decimals = range_to_decimals(
            self._layer.contrast_limits_range, self._layer.dtype
        )
        self.contrast_limits_slider.setRange(
            *self._layer.contrast_limits_range
        )
        self.contrast_limits_slider.setSingleStep(10**-decimals)
        self.contrast_limits_slider.setValue(self._layer.contrast_limits)
        self.contrast_limits_slider.setToolTip(
            trans._('Right click for detailed slider popup.')
        )

        self.clim_popup = None

        connect_setattr(
            self.contrast_limits_slider.valueChanged,
            self._layer,
            'contrast_limits',
        )
        connect_setattr(
            self.contrast_limits_slider.rangeChanged,
            self._layer,
            'contrast_limits_range',
        )

        self.contrast_limits_slider_label = QtWrappedLabel(
            trans._('contrast limits:')
        )

    def show_clim_popup(self):
        self.clim_popup = QContrastLimitsPopup(self._layer, self.parent())
        self.clim_popup.setParent(self.parent())
        self.clim_popup.move_to('top', min_length=650)
        self.clim_popup.show()

    def _on_contrast_limits_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrast_limits_slider):
            self.contrast_limits_slider.setValue(self._layer.contrast_limits)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setValue(self._layer.contrast_limits)

    def _on_contrast_limits_range_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrast_limits_slider):
            decimals = range_to_decimals(
                self._layer.contrast_limits_range, self._layer.dtype
            )
            self.contrast_limits_slider.setRange(
                *self._layer.contrast_limits_range
            )
            self.contrast_limits_slider.setSingleStep(10**-decimals)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setRange(
                    *self._layer.contrast_limits_range
                )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.auto_scale_bar_label, self.auto_scale_bar),
            (self.contrast_limits_slider_label, self.contrast_limits_slider),
        ]
