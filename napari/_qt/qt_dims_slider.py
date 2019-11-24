from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QLabel,
    QSpinBox,
)
from qtpy.QtGui import QFont, QFontMetrics

from .qt_scrollbar import ModifiedScrollBar
from .qt_modal import ModalPopup


class DimSliderWidget(QWidget):
    """Compound widget to hold the label, slider and play button for an axis.

    These will usually be instantiated in the QtDims._create_sliders method.
    """

    label_changed = Signal(int, str)  # axis, label

    def __init__(self, axis: int, parent: QWidget = None):
        super().__init__(parent=parent)
        self.axis = axis
        self.dims = parent.dims
        layout = QHBoxLayout()
        self._create_axis_label_widget()
        self._create_range_slider_widget()
        self._create_play_button_widget()
        layout.addWidget(self.label)
        layout.addWidget(self.play_button)
        layout.addWidget(self.slider, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.setLayout(layout)

    def _create_play_button_widget(self):
        # rbutton = QtPlayButton(self.parent(), self.axis, True)
        self.play_button = QtPlayButton(self.parent(), self.axis)

    def _update_label(self):
        """When any of the labels get updated, we want all of the label widths
        to be updated to the width of the longest label.  This keeps the
        sliders left-aligned.  This allows the full label to be visible at all
        times, without setting stretch on the layout.
        """
        with self.dims.events.axis_labels.blocker():
            self.dims.set_axis_label(self.axis, self.label.text())
        fm = QFontMetrics(QFont("", 0))
        labels = self.parent().findChildren(QLineEdit, 'axis_label')
        maxwidth = max([fm.width(lab.text()) for lab in labels])
        for labl in labels:
            labl.setFixedWidth(maxwidth + 10)
        self.label.clearFocus()
        self.parent().setFocus()

    def _create_axis_label_widget(self):
        """Create the axis label widget which accompanies its slider.

        Returns
        -------
        label : QLabel
            A label with the given text
        """
        label = QLineEdit(self)
        label.setObjectName('axis_label')  # needed for _update_label
        label.setText(self.dims.axis_labels[self.axis])
        label.home(False)
        label.setToolTip('Type to change axis label')
        label.setAcceptDrops(False)
        label.setEnabled(True)
        label.setAlignment(Qt.AlignRight)
        label.setContentsMargins(0, 0, 2, 0)
        label.editingFinished.connect(self._update_label)
        self.label = label
        self._update_label()

    def _create_range_slider_widget(self):
        """Creates a range slider widget for a given axis."""
        _range = self.dims.range[self.axis]
        # Set the maximum values of the range slider to be one step less than
        # the range of the layer as otherwise the slider can move beyond the
        # shape of the layer as the endpoint is included
        _range = (_range[0], _range[1] - _range[2], _range[2])
        point = self.dims.point[self.axis]

        slider = ModifiedScrollBar(Qt.Horizontal)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(_range[0])
        slider.setMaximum(_range[1])
        slider.setSingleStep(_range[2])
        slider.setPageStep(_range[2])
        slider.setValue(point)

        # Listener to be used for sending events back to model:
        slider.valueChanged.connect(
            lambda value: self.dims.set_point(self.axis, value)
        )

        def slider_focused_listener():
            self.parent().last_used = self.axis

        # linking focus listener to the last used:
        slider.sliderPressed.connect(slider_focused_listener)
        self.slider = slider


class QtPlayButton(QPushButton):
    """Play button, included in the DimSliderWidget, to control playback"""

    def __init__(self, dims, axis, reverse=False, fps=10):
        super().__init__()
        self.dims = dims
        self.axis = axis
        self.reverse = reverse
        self.fps = fps
        self.setProperty('reverse', str(reverse))  # for styling
        self.setProperty('playing', 'False')  # for styling

        dims.play_started.connect(self._handle_start)
        dims.play_stopped.connect(self._handle_stop)

        self.popup = ModalPopup(self)
        fpsspin = QSpinBox(self.popup)
        fpsspin.setValue(self.fps)
        fpsspin.valueChanged.connect(self.set_fps)
        self.popup.form_layout.insertRow(
            0, QLabel('frame rate:', parent=self.popup), fpsspin
        )

        dimsrange = dims.dims.range[axis]
        minspin = QSpinBox(self.popup)
        minspin.setValue(dimsrange[0])
        # minspin.valueChanged.connect(self.set_fps)
        self.popup.form_layout.insertRow(
            1, QLabel('start frame:', parent=self.popup), minspin
        )

        maxspin = QSpinBox(self.popup)
        maxspin.setValue(dimsrange[1] * dimsrange[2])
        # maxspin.valueChanged.connect(self.set_fps)
        self.popup.form_layout.insertRow(
            2, QLabel('end frame:', parent=self.popup), maxspin
        )

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.popup.show_above_mouse)
        self.clicked.connect(self._on_click)

    def set_fps(self, value):
        self.fps = int(value)

    def _handle_start(self, axis, fps):
        if (axis == self.axis) and (fps < 0) == self.reverse:
            self.setProperty('playing', 'True')
            self.style().unpolish(self)
            self.style().polish(self)

    def _handle_stop(self):
        self.setProperty('playing', 'False')
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_click(self):
        if self.property('playing') == "True":
            return self.dims.stop()
        self.dims.play(self.axis, self.fps * (-1 if self.reverse else 1))
