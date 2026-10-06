from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import (
    QDoubleRangeSlider,
    QLabeledDoubleRangeSlider,
    QLabeledDoubleSlider,
)

from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers import Layer
from napari.layers.intensity_mixin import IntensityVisualizationMixin
from napari.utils._dtype import normalize_dtype
from napari.utils.events.event_utils import connect_no_arg, connect_setattr


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
    """Popup for contrast limits with histogram visualization."""

    def __init__(
        self,
        layer: IntensityVisualizationMixin,
        parent: Optional[QWidget] = None,
        contrast_control: Optional[QtContrastLimitsControl] = None,
    ) -> None:
        super().__init__(parent)

        self._layer = layer
        self._contrast_control = contrast_control
        self._cleaned_up = False
        self._frame_base_height = 0

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(6)
        self.frame.setLayout(self._layout)

        # 1. Contrast limits slider
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

        clim_row = QHBoxLayout()
        clim_row.setContentsMargins(0, 0, 0, 0)
        clim_row.addWidget(QLabel('contrast limits:'))
        clim_row.addWidget(self.slider)
        self._layout.addLayout(clim_row)

        QApplication.processEvents()
        self.slider._reposition_labels()

        connect_setattr(self.slider.valueChanged, layer, 'contrast_limits')
        connect_setattr(
            self.slider.rangeChanged, layer, 'contrast_limits_range'
        )

        # 2. Gamma slider
        self.gamma_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.gamma_slider.setMinimum(0.2)
        self.gamma_slider.setMaximum(2.0)
        self.gamma_slider.setSingleStep(0.02)
        self.gamma_slider.setValue(layer.gamma)
        self.gamma_slider.setToolTip('Adjust gamma correction (0.2 - 2.0)')
        connect_setattr(self.gamma_slider.valueChanged, layer, 'gamma')
        connect_setattr(layer.events.gamma, self.gamma_slider, 'setValue')

        gamma_row = QHBoxLayout()
        gamma_row.setContentsMargins(0, 0, 0, 0)
        gamma_row.addWidget(QLabel('gamma:'))
        gamma_row.addWidget(self.gamma_slider)
        self._layout.addLayout(gamma_row)

        # 3. Reset / full range buttons
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

        reset_btn = QPushButton('reset')
        reset_btn.setObjectName('reset_clims_button')
        reset_btn.setToolTip('Autoscale contrast to data range')
        reset_btn.setFixedWidth(45)
        reset_btn.clicked.connect(reset)
        button_layout.addWidget(reset_btn)

        # the "full range" button doesn't do anything if it's not an
        # unsigned integer type (it's unclear what range should be set)
        # so we don't show create it at all.
        if np.issubdtype(normalize_dtype(layer.dtype), np.integer):
            range_btn = QPushButton('full range')
            range_btn.setObjectName('full_clim_range_button')
            range_btn.setToolTip('Set contrast range to full bit-depth')
            range_btn.setFixedWidth(75)
            range_btn.clicked.connect(layer.reset_contrast_limits_range)
            button_layout.addWidget(range_btn)

        # QtHistogramContentWidget full construction (vispy canvas) is deferred to
        # _schedule_histogram_compute() to avoid a PySide6 segfault when creating
        # native GL widgets during __init__.
        if self._contrast_control is not None:
            self.histogram_content = QtHistogramContentWidget(self._layer)
        self._layout.addWidget(self.histogram_content)
        self.histogram_button = QtModePushButton(
            self._layer,
            'histogram',
            tooltip=('Left click to toggle histogram visualization.\n'),
        )
        self.histogram_button.setCheckable(True)
        self.histogram_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.histogram_button.toggled.connect(
            self._on_histogram_button_toggled
        )
        button_layout.addWidget(self.histogram_button)

        button_layout.addStretch()
        buttons = QWidget()
        buttons.setLayout(button_layout)
        self._layout.addWidget(buttons)

        # Capture frame height WITHOUT histogram (baseline)
        self._layout.activate()
        self._frame_base_height = self.frame.sizeHint().height()

    def keyPressEvent(self, event):
        """Move focus to the slider when return is pressed."""
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.slider.setFocus()
            return
        super().keyPressEvent(event)

    def _base_height(self) -> int:
        """Popup height without the histogram widget."""
        outer = self.layout().contentsMargins()
        return self._frame_base_height + outer.top() + outer.bottom()

    def _on_histogram_button_toggled(self, visible: bool) -> None:
        """Handle left-click on histogram button to toggle histogram widget."""
        if self._contrast_control is None:
            # should only happen in testing
            return

        if visible:
            self.histogram_content._schedule_histogram_compute()
            self.histogram_content.show()
            # process events to ensure the size hint is up to date
            QApplication.processEvents()
            h = self.histogram_content.sizeHint().height()
            self.setFixedHeight(
                self._base_height() + h + self._layout.spacing()
            )
        else:
            self.histogram_content.hide()
            self.setFixedHeight(self._base_height())
            self.histogram_content._abort_histogram_worker()


class AutoScaleButtons(QWidget):
    def __init__(
        self,
        layer: IntensityVisualizationMixin,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)

        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.once_btn = QPushButton('once')
        self.once_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.auto_btn = QPushButton('continuous')
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(layer.auto_contrast)
        self.auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.once_btn.clicked.connect(lambda: self.auto_btn.setChecked(False))
        connect_no_arg(self.once_btn.clicked, layer, 'reset_contrast_limits')
        connect_setattr(self.auto_btn.toggled, layer, 'auto_contrast')

        self.layout().addWidget(self.once_btn)
        self.layout().addWidget(self.auto_btn)


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
    auto_scale_buttons : AutoScaleButtons
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    auto_scale_buttons_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the auto-contrast functionality widget.
    clim_popup : QContrastLimitsPopup
        Popup widget launching the contrast range slider.
    contrast_limits_slider : _QDoubleRangeSlider
        Slider controlling current constrast limits of the layer.
    contrast_limits_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the constrast limits chooser widget.
    """

    _layer: IntensityVisualizationMixin & Layer

    def __init__(
        self, parent: QWidget, layer: IntensityVisualizationMixin
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self._layer.events.contrast_limits_range.connect(
            self._on_contrast_limits_range_change
        )
        self._layer.events.auto_contrast.connect(self._on_auto_contrast_change)

        # Setup widgets
        self.auto_scale_buttons = AutoScaleButtons(layer, parent)
        self.auto_scale_buttons_label = QtWrappedLabel('auto-contrast:')
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
            'Right click for detailed slider popup.'
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

        self.contrast_limits_slider_label = QtWrappedLabel('contrast limits:')

        # the _clim_row is a wrapper around the contrast limits slider and the
        # histogram toggle button, so the sit on the same row
        self._clim_row = QFrame()
        self._clim_row.setFrameShape(QFrame.Shape.NoFrame)
        self._clim_row.setStyleSheet('QFrame { background: transparent; }')
        self._clim_layout = QHBoxLayout()
        self._clim_layout.setContentsMargins(0, 0, 0, 0)
        self._clim_layout.setSpacing(2)
        self._clim_layout.addWidget(self.contrast_limits_slider)
        self._clim_row.setLayout(self._clim_layout)

        self.histogram_button = QtModePushButton(
            self._layer,
            'histogram',
            tooltip=(
                'Left click to toggle histogram visualization.\n'
                'Right click to open histogram and contrast limits popup.'
            ),
        )
        self.histogram_button.setCheckable(True)
        self.histogram_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.histogram_button.toggled.connect(
            self._on_histogram_button_toggled
        )
        self.histogram_button.installEventFilter(self)
        self._clim_layout.addWidget(self.histogram_button)

        # empty wrapper, will be populated on first toggle (otherwise
        # it may segfault in some cases)
        self.histogram_content = QtHistogramContentWidget(self._layer)
        self.histogram_content.setProperty('foreground', 'true')
        self.histogram_content.hide()

    def show_clim_popup(self):
        self.clim_popup = QContrastLimitsPopup(
            layer=self._layer,
            parent=self.parent(),
            contrast_control=self,
        )
        if self.parent():
            self.clim_popup.move_to('top', min_length=650)
        self.clim_popup.show()

    def _on_contrast_limits_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrast_limits_slider):
            self.contrast_limits_slider.setValue(self._layer.contrast_limits)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setValue(self._layer.contrast_limits)

    def _on_auto_contrast_change(self):
        """Receive layer model auto_contrast change event and update buttons."""
        with qt_signals_blocked(self.auto_scale_buttons.auto_btn):
            self.auto_scale_buttons.auto_btn.setChecked(
                self._layer.auto_contrast
            )

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

    def eventFilter(self, obj, event):
        """Handle right-click on histogram button to show popup."""
        if (
            self.histogram_button is not None
            and obj == self.histogram_button
            and event.type() == event.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.histogram_button.setDown(False)
            self.show_clim_popup()
            return True
        return super().eventFilter(obj, event)

    def _on_histogram_button_toggled(self, visible: bool) -> None:
        """Handle left-click on histogram button to toggle histogram widget."""
        if visible:
            self.histogram_content._schedule_histogram_compute()
            self.histogram_content.show()
        else:
            self.histogram_content.hide()
            self.histogram_content._abort_histogram_worker()

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [
            (self.auto_scale_buttons_label, self.auto_scale_buttons),
            (self.contrast_limits_slider_label, self._clim_row),
            (self.histogram_content,),
        ]
