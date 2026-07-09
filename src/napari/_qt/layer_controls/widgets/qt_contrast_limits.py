from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
from napari.layers import Image, Surface
from napari.utils._dtype import normalize_dtype
from napari.utils.events import disconnect_events
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
        self,
        layer: Image | Surface,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._layer = layer
        self._cleaned_up = False

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

        # 4. Reset / full range buttons
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
            range_btn.setToolTip(
                trans._('Set contrast range to full bit-depth')
            )
            range_btn.setFixedWidth(75)
            range_btn.clicked.connect(layer.reset_contrast_limits_range)
            button_layout.addWidget(range_btn)

        # Histogram toggle checkbox (Image layers only)
        if isinstance(layer, Image):
            self._histogram_enabled_checkbox = QCheckBox('histogram')
            self._histogram_enabled_checkbox.setChecked(
                layer.histogram.enabled
            )
            self._histogram_enabled_checkbox.setToolTip(
                'Show histogram in this popup'
            )
            self._histogram_enabled_checkbox.toggled.connect(
                self._on_popup_histogram_toggled
            )
            button_layout.addWidget(self._histogram_enabled_checkbox)

        button_layout.addStretch()

        self._layout.addWidget(self._create_widget_from_layout(button_layout))

        # Histogram content — created after all non-histogram layout items
        # so _frame_base_height captures the true baseline (clim + gamma + buttons).
        self.histogram_content = None
        self._histogram_enabled_checkbox = None
        self._frame_base_height: int = 0
        if isinstance(layer, Image):
            self.histogram_content = QtHistogramContentWidget(
                layer,
                parent=self,
            )
            # Capture frame height with all non-histogram items in place
            self._layout.activate()
            self._frame_base_height = self.frame.sizeHint().height()
            # Insert between clim row (0) and gamma row (now index 1)
            self._layout.insertWidget(1, self.histogram_content)
            if not layer.histogram.enabled:
                self.histogram_content.setFixedHeight(0)
                self.histogram_content.hide()
            layer.histogram.events.enabled.connect(
                self._on_external_histogram_enabled
            )

    def keyPressEvent(self, event):
        """Override to prevent Enter from closing the popup.

        Hitting Enter/Return inside the popup should defocus the
        line-edit rather than closing the window, consistent with
        the behaviour of the original QRangeSliderPopup.
        """
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.slider.setFocus()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up on close to prevent event-listener leaks."""
        self._cleanup()
        super().closeEvent(event)

    def sizeHint(self):
        """Return the preferred size, excluding the histogram when hidden.

        ``move_to`` calls ``sizeHint()`` to set the popup's initial geometry,
        but ``QLayout.sizeHint()`` includes hidden widgets.  We override so
        the initial height is correct regardless of histogram visibility.
        """
        hint = super().sizeHint()
        if self.histogram_content and self.histogram_content.isHidden():
            outer = self.layout().contentsMargins()
            hint.setHeight(
                self._frame_base_height + outer.top() + outer.bottom()
            )
        return hint

    def _base_height(self) -> int:
        """Popup height without the histogram widget."""
        outer = self.layout().contentsMargins()
        return self._frame_base_height + outer.top() + outer.bottom()

    def _on_popup_histogram_toggled(self, visible: bool) -> None:
        """Handle the popup's histogram checkbox toggle."""
        if self.histogram_content is None:
            return
        if visible:
            h = self.histogram_content.sizeHint().height()
            self.histogram_content.setFixedHeight(h)
            self.histogram_content.show()
            self._layer.histogram.enabled = True
            self.setFixedHeight(
                self._base_height() + h + self._layout.spacing()
            )
        else:
            self.histogram_content.setFixedHeight(0)
            self.histogram_content.hide()
            self._layer.histogram.enabled = False
            self.setFixedHeight(self._base_height())

    def _on_external_histogram_enabled(self) -> None:
        """Sync checkbox when ``layer.histogram.enabled`` changes from outside."""
        if self._histogram_enabled_checkbox is not None:
            with qt_signals_blocked(self._histogram_enabled_checkbox):
                self._histogram_enabled_checkbox.setChecked(
                    self._layer.histogram.enabled
                )
            if self.histogram_content is not None:
                if self._layer.histogram.enabled:
                    h = self.histogram_content.sizeHint().height()
                    self.histogram_content.setFixedHeight(h)
                    self.histogram_content.show()
                    self.setFixedHeight(
                        self._base_height() + h + self._layout.spacing()
                    )
                else:
                    self.histogram_content.setFixedHeight(0)
                    self.histogram_content.hide()
                    self.setFixedHeight(self._base_height())

    def _cleanup(self) -> None:
        """Disconnect event handlers and clean up widgets."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if isinstance(self._layer, Image):
            self._layer.histogram.events.enabled.disconnect(
                self._on_external_histogram_enabled
            )
        # Clear fixed-height constraint so the popup doesn't persist it
        self.setMaximumHeight(16777215)
        self.setMinimumHeight(0)
        if self.histogram_content is not None:
            self.histogram_content.cleanup()
            self.histogram_content = None

    def _create_widget_from_layout(self, layout: QHBoxLayout) -> QWidget:
        """Helper to wrap a layout in a widget."""
        widget = QWidget()
        widget.setLayout(layout)
        return widget


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

        # Wrap the slider (and optional histogram button) in a QFrame so
        # they sit on the same row in the form layout.  The QFrame is
        # created once here and reused in get_widget_controls() — creating
        # a new QFrame every time would reparent the slider, destroying the
        # C++ object when the temporary QFrame is collected.
        self._clim_row = QFrame()
        self._clim_row.setFrameShape(QFrame.Shape.NoFrame)
        self._clim_row.setStyleSheet('QFrame { background: transparent; }')
        self._clim_layout = QHBoxLayout()
        self._clim_layout.setContentsMargins(0, 0, 0, 0)
        self._clim_layout.setSpacing(2)
        self._clim_layout.addWidget(self.contrast_limits_slider)
        self._clim_row.setLayout(self._clim_layout)

        # Histogram toggle button — added alongside the slider via a
        # wrapper widget in get_widget_controls().
        self.histogram_button = None
        if isinstance(layer, Image):
            self.histogram_button = QtModePushButton(
                self._layer,
                'histogram',
                tooltip=(
                    'Left click to toggle histogram in layer controls.\n'
                    'Right click to open histogram popup.'
                ),
            )
            self.histogram_button.setCheckable(True)
            self.histogram_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.histogram_button.toggled.connect(
                self._on_histogram_button_toggled
            )
            self.histogram_button.installEventFilter(self)
            self._clim_layout.addWidget(self.histogram_button)

            # Sync button checked state when ``enabled`` changes via the API
            layer.histogram.events.enabled.connect(
                self._on_histogram_model_enabled
            )

    def show_clim_popup(self):
        self.clim_popup = QContrastLimitsPopup(
            self._layer,
            self.parent(),
        )
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

    def eventFilter(self, obj, event):
        """Handle right-click on histogram button to show popup."""
        if (
            self.histogram_button is not None
            and obj == self.histogram_button
            and event.type() == event.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.histogram_button.setDown(False)
            self.show_histogram_popup()
            return True
        return super().eventFilter(obj, event)

    def _on_histogram_button_toggled(self, visible: bool) -> None:
        """Handle left-click on histogram button to toggle histogram widget."""
        if not isinstance(self._layer, Image):
            return

        parent = self.parent()
        histogram_control = getattr(parent, '_histogram_control', None)
        if histogram_control is None:
            return

        histogram_control.ensure_content()

        # Show or hide the persistent content_widget; it is always present
        # in the form layout (inserted once by QtBaseImageControls) so we
        # never need to search layout rows.
        if visible:
            # Restore size policy so the form layout allocates space
            histogram_control.content_widget.show()
            histogram_control.content_widget.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred,
            )
            # Enable histogram computation; _on_enabled_change triggers
            # an immediate compute if there is pending dirty data.
            self._layer.histogram.enabled = True
        else:
            # Set size policy to Ignored so the form layout collapses the
            # column and doesn't reserve space for the hidden widget
            histogram_control.content_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Ignored,
            )
            histogram_control.content_widget.hide()
            # Disable histogram computation
            self._layer.histogram.enabled = False

    def show_histogram_popup(self):
        """Show the histogram popup widget."""
        if not isinstance(self._layer, Image):
            return

        # The popup's showEvent manages histogram enable/disable; do not
        # pre-enable here, or the popup cannot tell whether it was the one
        # that enabled it and will skip the matching disable on close.
        self.show_clim_popup()

    def _on_histogram_model_enabled(self) -> None:
        """Sync button checked state when ``layer.histogram.enabled`` changes via the API.

        Uses ``qt_signals_blocked`` so the ``toggled`` signal does NOT fire,
        preventing recursion into ``_on_histogram_button_toggled`` (which would
        re-set ``layer.histogram.enabled`` and re-dispatch the event).
        """
        if self.histogram_button is None or not isinstance(self._layer, Image):
            return
        with qt_signals_blocked(self.histogram_button):
            self.histogram_button.setChecked(self._layer.histogram.enabled)

    def disconnect_widget_controls(self) -> None:
        """Disconnect histogram model events and base controls."""
        if isinstance(self._layer, Image):
            disconnect_events(self._layer.histogram.events, self)
        super().disconnect_widget_controls()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.auto_scale_bar_label, self.auto_scale_bar),
            (self.contrast_limits_slider_label, self._clim_row),
        ]
