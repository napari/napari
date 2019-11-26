from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QLabel,
    QSpinBox,
    QComboBox,
)

from .qt_scrollbar import ModifiedScrollBar
from .qt_modal import QtModalPopup


class QtDimSliderWidget(QWidget):
    """Compound widget to hold the label, slider and play button for an axis.

    These will usually be instantiated in the QtDims._create_sliders method.
    """

    label_changed = Signal(int, str)  # axis, label

    def __init__(self, axis: int, parent: QWidget = None):
        super().__init__(parent=parent)
        self.axis = axis
        self.dims = parent.dims
        self.label = None
        self.slider = None
        self.play_button = None
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

        self.dims.events.axis_labels.connect(self._pull_label)

    def _create_play_button_widget(self):
        # rbutton = QtPlayButton(self.parent(), self.axis, True)
        self.play_button = QtPlayButton(self.parent(), self.axis)

    def _pull_label(self, event):
        if event.axis == self.axis:
            label = self.dims.axis_labels[self.axis]
            self.label.setText(label)
            self.label_changed.emit(self.axis, label)

    def _update_label(self):
        with self.dims.events.axis_labels.blocker():
            self.dims.set_axis_label(self.axis, self.label.text())
        self.label.clearFocus()
        self.parent().setFocus()
        self.label_changed.emit(self.axis, self.label.text())

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

    def _update_range(self):
        """Updates range for slider."""
        displayed_sliders = self.parent()._displayed_sliders

        _range = self.dims.range[self.axis]
        _range = (_range[0], _range[1] - _range[2], _range[2])
        if _range not in (None, (None, None, None)):
            if _range[1] == 0:
                displayed_sliders[self.axis] = False
                self.parent().last_used = None
                self.slider.hide()
            else:
                if (
                    not displayed_sliders[self.axis]
                    and self.axis not in self.dims.displayed
                ):
                    displayed_sliders[self.axis] = True
                    self.last_used = self.axis
                    self.slider.show()
                self.slider.setMinimum(_range[0])
                self.slider.setMaximum(_range[1])
                self.slider.setSingleStep(_range[2])
                self.slider.setPageStep(_range[2])
        else:
            displayed_sliders[self.axis] = False
            self.slider.hide()


class QtPlayButton(QPushButton):
    """Play button, included in the DimSliderWidget, to control playback"""

    play_requested = Signal(int)  # axis, fps

    def __init__(self, dims, axis, reverse=False, fps=10, mode='loop'):
        super().__init__()
        self.dims = dims
        self.axis = axis
        self.reverse = reverse
        self.fps = fps
        self.mode = mode
        self.setProperty('reverse', str(reverse))  # for styling
        self.setProperty('playing', 'False')  # for styling

        dims.play_started.connect(self._handle_start)
        dims.play_stopped.connect(self._handle_stop)

        # build popup modal form
        self.popup = QtModalPopup(self)
        self.fpsspin = QSpinBox(self.popup)
        self.fpsspin.setAlignment(Qt.AlignCenter)
        self.fpsspin.setValue(self.fps)
        self.fpsspin.setMaximum(500)
        self.fpsspin.setMinimum(-500)
        self.fpsspin.valueChanged.connect(self.set_fps)
        self.popup.form_layout.insertRow(
            0, QLabel('frames per sec:', parent=self.popup), self.fpsspin
        )

        # dimsrange = dims.dims.range[axis]
        # minspin = QSpinBox(self.popup)
        # minspin.setAlignment(Qt.AlignCenter)
        # minspin.setValue(dimsrange[0])
        # minspin.valueChanged.connect(self.set_minframe)
        # self.popup.form_layout.insertRow(
        #     1, QLabel('start frame:', parent=self.popup), minspin
        # )

        # maxspin = QSpinBox(self.popup)
        # maxspin.setAlignment(Qt.AlignCenter)
        # maxspin.setValue(dimsrange[1] * dimsrange[2])
        # maxspin.valueChanged.connect(self.set_maxframe)
        # self.popup.form_layout.insertRow(
        #     2, QLabel('end frame:', parent=self.popup), maxspin
        # )

        self.mode_combo = QComboBox(self.popup)
        self.mode_combo.addItems(['loop', 'back and forth', 'play once'])
        self.popup.form_layout.insertRow(
            1, QLabel('play mode:', parent=self.popup), self.mode_combo
        )
        self.mode_combo.currentTextChanged.connect(self.set_mode)
        self.mode_combo.setCurrentText(self.mode)
        self.clicked.connect(self._on_click)

    def mouseReleaseEvent(self, event):
        # using this instead of self.customContextMenuRequested.connect
        # because the latter was not sending the rightMouseButton
        # release event.
        if event.button() == Qt.RightButton:
            self.popup.show_above_mouse()
        elif event.button() == Qt.LeftButton:
            self._on_click()

    def set_fps(self, value):
        print('setfps')
        if value != 0:
            self.fps = value
            self.fpsspin.setValue(value)
        if self.dims.is_playing:
            self.play_requested.emit(self.axis)

    def set_mode(self, value):
        self.mode_combo.setCurrentText(value)
        text = self.mode_combo.currentText()
        if text.startswith('back'):
            self.mode = 'loop_back_and_forth'
        elif text.startswith('play'):
            self.mode = 'once'
        else:
            self.mode = 'loop'
        if self.dims.is_playing:
            self.play_requested.emit(self.axis)

    def _handle_start(self, axis, fps):
        if (axis == self.axis) and fps != 0:
            print("START")
            self.setProperty('playing', 'True')
            self.style().unpolish(self)
            self.style().polish(self)

    def _handle_stop(self):
        print("STOP")
        self.setProperty('playing', 'False')
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_click(self):
        if self.property('playing') == "True":
            return self.dims.stop()
        self.play_requested.emit(self.axis)
