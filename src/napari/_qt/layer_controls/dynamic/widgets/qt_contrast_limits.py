from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
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
from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.qthreading import GeneratorWorker, create_worker
from napari._qt.utils import (
    qt_signals_blocked,
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.utils._dtype import normalize_dtype
from napari.utils.events import disconnect_events
from napari.utils.events.event_utils import connect_no_arg, connect_setattr

if TYPE_CHECKING:
    from napari.layers import Image, Surface

_COMPUTE_DEBOUNCE_MS = 50


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
        layers: list[Image | Surface],
        parent: Optional[QWidget] = None,
        contrast_control=None,
    ) -> None:
        super().__init__(parent)

        self._layers = layers
        self._contrast_control = contrast_control
        self._cleaned_up = False
        self._histogram_enabled_checkbox = None

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

        decimals = range_to_decimals(
            self._layers[0].contrast_limits_range, self._layers[0].dtype
        )
        self.slider.setRange(*self._layers[0].contrast_limits_range)
        self.slider.setDecimals(decimals)
        self.slider.setSingleStep(10**-decimals)
        self.slider.setValue(self._layers[0].contrast_limits)

        clim_row = QHBoxLayout()
        clim_row.setContentsMargins(0, 0, 0, 0)
        clim_row.addWidget(QLabel('contrast limits:'))
        clim_row.addWidget(self.slider)
        self._layout.addLayout(clim_row)

        QApplication.processEvents()
        self.slider._reposition_labels()

        for layer in self._layers:
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
        self.gamma_slider.setValue(self._layers[0].gamma)
        self.gamma_slider.setToolTip('Adjust gamma correction (0.2 - 2.0)')
        for layer in self._layers:
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

        reset_btn = QPushButton('reset')
        reset_btn.setObjectName('reset_clims_button')
        reset_btn.setToolTip('Autoscale contrast to data range')
        reset_btn.setFixedWidth(45)
        reset_btn.clicked.connect(self._reset)
        button_layout.addWidget(reset_btn)

        # the "full range" button doesn't do anything if it's not an
        # unsigned integer type (it's unclear what range should be set)
        # so we don't show create it at all.
        if all(
            np.issubdtype(normalize_dtype(layer.dtype), np.integer)
            for layer in self._layers
        ):
            range_btn = QPushButton('full range')
            range_btn.setObjectName('full_clim_range_button')
            range_btn.setToolTip('Set contrast range to full bit-depth')
            range_btn.setFixedWidth(75)
            for layer in self._layers:
                range_btn.clicked.connect(layer.reset_contrast_limits_range)
            button_layout.addWidget(range_btn)

        # Histogram toggle checkbox (single Image/Surface layer only).  The
        # checkbox itself is always created (simple Qt widget, safe), but the
        # QtHistogramContentWidget (vispy canvas) is deferred to
        # _ensure_histogram_content() to avoid a PySide6 segfault when
        # creating native GL widgets during __init__.
        if len(self._layers) == 1:
            self.histogram_content = None
            self._frame_base_height: int = 0

            self._histogram_enabled_checkbox = QCheckBox('histogram')
            self._histogram_enabled_checkbox.setChecked(False)
            self._histogram_enabled_checkbox.setToolTip(
                'Show histogram in this popup'
            )
            self._histogram_enabled_checkbox.toggled.connect(
                self._on_popup_histogram_toggled
            )
            button_layout.addWidget(self._histogram_enabled_checkbox)
            self._needs_content_on_show = False

        button_layout.addStretch()

        self._layout.addWidget(self._create_widget_from_layout(button_layout))

        # Capture frame height WITHOUT histogram (baseline)
        self._layout.activate()
        self._frame_base_height = self.frame.sizeHint().height()

    def _reset(self):
        for layer in self._layers:
            layer.reset_contrast_limits()
            layer.contrast_limits_range = layer.contrast_limits
            decimals_ = range_to_decimals(
                layer.contrast_limits_range, layer.dtype
            )
            self.slider.setDecimals(decimals_)
            self.slider.setSingleStep(10**-decimals_)
            self.slider.setRange(*layer.contrast_limits_range)

    def showEvent(self, event):
        """Create histogram content lazily on first show to avoid PySide6
        segfault during __init__ when vispy native widgets are created."""
        super().showEvent(event)
        if getattr(self, '_needs_content_on_show', False):
            self._needs_content_on_show = False
            self._ensure_histogram_content()

    def keyPressEvent(self, event):
        """Move focus to the slider when return is pressed."""
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.slider.setFocus()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up on close to prevent event-listener leaks."""
        self._cleanup()
        super().closeEvent(event)

    def _cleanup(self) -> None:
        """Disconnect event handlers and clean up widgets."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if len(self._layers) == 1 and self.histogram_content is not None:
            self.histogram_content.cleanup()
            self.histogram_content = None

    def _base_height(self) -> int:
        """Popup height without the histogram widget."""
        outer = self.layout().contentsMargins()
        return self._frame_base_height + outer.top() + outer.bottom()

    def _ensure_histogram_content(self) -> None:
        """Lazy-create the histogram content widget.

        Called once, either from showEvent (if histogram was enabled before
        the popup was opened) or from _set_histogram_visible (if the user
        checks the checkbox after the popup is already visible).
        """
        if self.histogram_content is not None:
            return

        self.histogram_content = QtHistogramContentWidget(
            self._layers[0],
            parent=self,
        )
        self._layout.insertWidget(1, self.histogram_content)
        self.histogram_content.hide()

    def _set_histogram_visible(self, visible: bool) -> None:
        """Show or hide the histogram content and resize the popup."""
        self._ensure_histogram_content()
        if self.histogram_content is None:
            return
        if visible:
            h = self.histogram_content.sizeHint().height()
            self.histogram_content.show()
            if self._contrast_control is not None:
                self._contrast_control._schedule_compute()
            self.setFixedHeight(
                self._base_height() + h + self._layout.spacing()
            )
        else:
            self.histogram_content.hide()
            self.setFixedHeight(self._base_height())

    def _on_popup_histogram_toggled(self, visible: bool) -> None:
        """Handle the popup's histogram checkbox toggle."""
        self._set_histogram_visible(visible)

    def _create_widget_from_layout(self, layout: QHBoxLayout) -> QWidget:
        """Helper to wrap a layout in a widget."""
        widget = QWidget()
        widget.setLayout(layout)
        return widget


class AutoScaleButtons(QWidget):
    def __init__(
        self, layers: list[Image | Surface], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent=parent)

        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.once_btn = QPushButton('once')
        self.once_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.auto_btn = QPushButton('continuous')
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(layers[0].auto_contrast)
        self.auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.once_btn.clicked.connect(lambda: self.auto_btn.setChecked(False))
        for layer in layers:
            connect_no_arg(
                self.once_btn.clicked, layer, 'reset_contrast_limits'
            )
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
    layers : list[napari.layers.Image | napari.layers.Surface]
        A list of napari Image and Surface layers.

    Attributes
    ----------
    auto_scale_buttons : AutoScaleButtons
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    auto_scale_buttons_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the auto-contrast functionality widget.
    clim_popup : napari._qt.qt_range_slider_popup.QRangeSliderPopup
        Popup widget launching the contrast range slider.
    contrast_limits_slider : _QDoubleRangeSlider
        Slider controlling current constrast limits of the layer.
    contrast_limits_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the constrast limits chooser widget.
    """

    _layers: list[Image | Surface]

    def __init__(
        self, layers: list[Image | Surface], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.contrast_limits.connect(
                self._on_contrast_limits_change
            )
            layer.events.contrast_limits_range.connect(
                self._on_contrast_limits_range_change
            )
            layer.events.auto_contrast.connect(self._on_auto_contrast_change)
        # Setup widgets
        self.auto_scale_buttons = AutoScaleButtons(self._layers, parent)
        self.auto_scale_buttons_label = QtWrappedLabel('auto-contrast:')
        self.contrast_limits_slider = _QDoubleRangeSlider(
            Qt.Orientation.Horizontal,
        )
        self.contrast_limits_slider.show_clim_popup.connect(
            self.show_clim_popup
        )
        # set widget range and step size based on first layer since this is only display
        decimals = range_to_decimals(
            self._layers[0].contrast_limits_range, self._layers[0].dtype
        )
        self.contrast_limits_slider.setRange(
            *self._layers[0].contrast_limits_range
        )
        self.contrast_limits_slider.setSingleStep(10**-decimals)
        # set value of slider based on first layer until we implement a way to handle multiple layers
        self.contrast_limits_slider.setValue(self._layers[0].contrast_limits)
        self.contrast_limits_slider.setToolTip(
            'Right click for detailed slider popup.'
        )

        self.clim_popup = None

        for layer in self._layers:
            connect_setattr(
                self.contrast_limits_slider.valueChanged,
                layer,
                'contrast_limits',
            )
            connect_setattr(
                self.contrast_limits_slider.rangeChanged,
                layer,
                'contrast_limits_range',
            )

        self.contrast_limits_slider_label = QtWrappedLabel('contrast limits:')

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
        self.histogram_button = QtModePushButton(
            self._layers[0],
            'histogram',
        )
        self.histogram_button.setCheckable(True)
        self.histogram_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Histogram is only supported for a single selected layer; multiple
        # selected layers are disabled until we figure out how to handle them.
        self._histogram_single = len(self._layers) == 1
        if self._histogram_single:
            self.histogram_button.setToolTip(
                'Left click to toggle histogram in layer controls.\n'
                'Right click to open histogram popup.'
            )
            self.histogram_button.toggled.connect(
                self._on_histogram_button_toggled
            )
            self.histogram_button.installEventFilter(self)

            self._histogram_content_widget = QWidget()
            self._histogram_content_widget.hide()
            self._histogram_content_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
            )
            self._histogram_content = None
            self._content_layout = QVBoxLayout()
            self._content_layout.setContentsMargins(4, 4, 4, 4)
            self._content_layout.setSpacing(4)
            self._histogram_content_widget.setLayout(self._content_layout)

            self._compute_timer = QTimer()
            self._compute_timer.setSingleShot(True)
            self._compute_timer.timeout.connect(self._run_compute)
            self._worker: GeneratorWorker | None = None
            self._compute_epoch = 0

            layer = self._layers[0]
            for ev in (
                layer.histogram.events.bins,
                layer.histogram.events.max_samples,
                layer.histogram.events.mode,
                layer.histogram.events.log_scale,
            ):
                ev.connect(self._schedule_compute)
            for ev in (
                layer.events.data,
                layer.events.contrast_limits_range,
                layer.events.set_data,
            ):
                ev.connect(self._schedule_compute)
        else:
            self.histogram_button.setToolTip(
                'Histogram is currently only supported for a single selected '
                'layer, not for multiple selected layers.'
            )
            set_widgets_enabled_with_opacity(
                self, [self.histogram_button], False
            )

        self._clim_layout.addWidget(self.histogram_button)

    def show_clim_popup(self):
        self.clim_popup = QContrastLimitsPopup(
            layers=self._layers,
            parent=self.contrast_limits_slider.parent(),
            contrast_control=self,
        )
        self.clim_popup.move_to('top', min_length=650)
        self.clim_popup.show()

    def _on_contrast_limits_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrast_limits_slider):
            self.contrast_limits_slider.setValue(
                self._layers[0].contrast_limits
            )

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setValue(
                    self._layers[0].contrast_limits
                )

    def _on_auto_contrast_change(self):
        """Receive layer model auto_contrast change event and update buttons."""
        with qt_signals_blocked(self.auto_scale_buttons.auto_btn):
            self.auto_scale_buttons.auto_btn.setChecked(
                self._layers[0].auto_contrast
            )

    def _on_contrast_limits_range_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrast_limits_slider):
            decimals = range_to_decimals(
                self._layers[0].contrast_limits_range, self._layers[0].dtype
            )
            self.contrast_limits_slider.setRange(
                *self._layers[0].contrast_limits_range
            )
            self.contrast_limits_slider.setSingleStep(10**-decimals)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setRange(
                    *self._layers[0].contrast_limits_range
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
        if not self._histogram_single:
            return
        if visible:
            self.ensure_content()
            self._histogram_content_widget.show()
            self._histogram_content_widget.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
            )
            self._schedule_compute()
        else:
            self._histogram_content_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
            )
            self._histogram_content_widget.hide()

    def ensure_content(self) -> None:
        """Lazily create the histogram content widget (vispy canvas)."""
        if self._histogram_content is not None:
            return
        self._histogram_content = QtHistogramContentWidget(
            self._layers[0],
            parent=self._histogram_content_widget,
        )
        self._content_layout.addWidget(self._histogram_content)

    def _schedule_compute(self, event=None) -> None:
        """Debounce histogram recomputation via a single-shot timer."""
        self._compute_timer.start(_COMPUTE_DEBOUNCE_MS)

    def _run_compute(self) -> None:
        """Run the async histogram compute (single worker, epoch-guarded)."""
        if getattr(self, '_cleaned_up', False):
            return
        self._abort_worker()
        self._compute_epoch += 1
        epoch = self._compute_epoch
        layer = self._layers[0]
        worker = create_worker(layer.histogram.compute_async, layer)
        worker.yielded.connect(lambda bc: self._on_yield(bc, epoch))
        worker.finished.connect(lambda: self._on_worker_done(epoch))
        self._worker = worker
        worker.start()

    def _on_yield(self, bin_counts: tuple, epoch: int) -> None:
        """Write a progressive result and broadcast it to all views."""
        if getattr(self, '_cleaned_up', False) or epoch != self._compute_epoch:
            return
        bin_edges, counts = bin_counts
        self._layers[0].metadata['_computed_histogram'] = {
            'bin_edges': bin_edges,
            'counts': counts,
        }
        self._layers[0].histogram.events.updated()

    def _on_worker_done(self, epoch: int) -> None:
        """Finalize the compute broadcast once the worker finishes."""
        if epoch != self._compute_epoch:
            return
        self._worker = None
        self._layers[0].histogram.events.completed()

    def _abort_worker(self) -> None:
        """Stop any in-flight compute worker."""
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        worker.yielded.disconnect()
        worker.finished.disconnect()
        pbar = getattr(worker, 'pbar', None)
        if pbar is not None:
            pbar.close()
        worker.quit()

    def disconnect_widget_controls(self) -> None:
        """Disconnect histogram model events and base controls."""
        if self._histogram_single:
            self._cleaned_up = True
            self._compute_timer.stop()
            self._abort_worker()
        for layer in self._layers:
            disconnect_events(layer.histogram.events, self)
        super().disconnect_widget_controls()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        controls = [
            (self.auto_scale_buttons_label, self.auto_scale_buttons),
            (self.contrast_limits_slider_label, self._clim_row),
        ]
        if self._histogram_single:
            controls.append((self._histogram_content_widget,))
        return controls
