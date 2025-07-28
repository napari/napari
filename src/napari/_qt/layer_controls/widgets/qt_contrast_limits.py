from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from superqt import QDoubleRangeSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_range_slider_popup import QRangeSliderPopup
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


class QContrastLimitsPopup(QRangeSliderPopup):
    def __init__(
        self, layer: Image | Surface, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)

        decimals = range_to_decimals(layer.contrast_limits_range, layer.dtype)
        self.slider.setRange(*layer.contrast_limits_range)
        self.slider.setDecimals(decimals)
        self.slider.setSingleStep(10**-decimals)
        self.slider.setValue(layer.contrast_limits)

        connect_setattr(self.slider.valueChanged, layer, 'contrast_limits')
        connect_setattr(
            self.slider.rangeChanged, layer, 'contrast_limits_range'
        )

        def reset():
            layer.reset_contrast_limits()
            layer.contrast_limits_range = layer.contrast_limits
            decimals_ = range_to_decimals(
                layer.contrast_limits_range, layer.dtype
            )
            self.slider.setDecimals(decimals_)
            self.slider.setSingleStep(10**-decimals_)

        reset_btn = QPushButton('reset')
        reset_btn.setObjectName('reset_clims_button')
        reset_btn.setToolTip(trans._('Autoscale contrast to data range'))
        reset_btn.setFixedWidth(45)
        reset_btn.clicked.connect(reset)
        self._layout.addWidget(
            reset_btn, alignment=Qt.AlignmentFlag.AlignBottom
        )

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
            self._layout.addWidget(
                range_btn, alignment=Qt.AlignmentFlag.AlignBottom
            )


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
            (self.contrast_limits_slider_label, self.contrast_limits_slider),
            (self.auto_scale_bar_label, self.auto_scale_bar),
        ]
