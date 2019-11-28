from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal, Slot, QObject
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..util.event import Event
from .qt_modal import QtModalPopup
from .qt_scrollbar import ModifiedScrollBar
from .util import new_worker_qthread


class AnimationWorker(QObject):
    """A thread to keep the animation timer independent of the main event loop.

    This prevents mouseovers and other events from causing animation lag. See
    QtDims.play() for public-facing docstring.
    """

    frame_requested = Signal(int, int)  # axis, point
    finished = Signal()
    started = Signal()

    def __init__(self, slider):
        super().__init__()
        self.slider = slider
        self.dims = slider.dims
        self.axis = slider.axis
        self.loop_mode = slider.loop_mode
        slider.fps_changed.connect(self.set_fps)
        slider.mode_changed.connect(self.set_loop_mode)
        slider.range_changed.connect(self.set_frame_range)
        self.set_fps(self.slider.fps)
        self.set_frame_range(slider.frame_range)

        # after dims.set_point is called, it will emit a dims.events.axis()
        # we use this to update this threads current frame (in case it
        # was some other event that updated the axis)
        self.dims.events.axis.connect(self._on_axis_changed)
        self.current = max(self.dims.point[self.axis], self.min_point)
        self.current = min(self.current, self.max_point)
        self.timer = QTimer()

    @Slot()
    def work(self):
        # if loop_mode is once and we are already on the last frame,
        # return to the first frame... (so the user can keep hitting once)
        if self.loop_mode == 0:
            if self.step > 0 and self.current >= self.max_point - 1:
                self.frame_requested.emit(self.axis, self.min_point)
            elif self.step < 0 and self.current <= self.min_point + 1:
                self.frame_requested.emit(self.axis, self.max_point)
            self.timer.singleShot(self.interval, self.advance)
        else:
            # immediately advance one frame
            self.advance()
        self.started.emit()

    @Slot(int)
    def set_fps(self, fps):
        if fps == 0:
            return self.finish()
        self.step = 1 if fps > 0 else -1  # negative fps plays in reverse
        self.interval = 1000 / abs(fps)

    @Slot(tuple)
    def set_frame_range(self, frame_range):
        self.dimsrange = self.dims.range[self.axis]

        if frame_range is not None:
            if frame_range[0] >= frame_range[1]:
                raise ValueError("frame_range[0] must be <= frame_range[1]")
            if frame_range[0] < self.dimsrange[0]:
                raise IndexError("frame_range[0] out of range")
            if frame_range[1] * self.dimsrange[2] >= self.dimsrange[1]:
                raise IndexError("frame_range[1] out of range")
        self.frame_range = frame_range

        if self.frame_range is not None:
            self.min_point, self.max_point = self.frame_range
        else:
            self.min_point = 0
            self.max_point = int(
                np.floor(self.dimsrange[1] - self.dimsrange[2])
            )
        self.max_point += 1  # range is inclusive

    @Slot(int)
    def set_loop_mode(self, mode):
        self.loop_mode = mode

    def advance(self):
        """Advance the current frame in the animation.

        Takes dims scale into account and restricts the animation to the
        requested frame_range, if entered.
        """
        self.current += self.step * self.dimsrange[2]
        if self.current < self.min_point:
            if self.loop_mode == 2:  # 'loop_back_and_forth'
                self.step *= -1
                self.current = self.min_point + self.step * self.dimsrange[2]
            elif self.loop_mode == 1:  # 'loop'
                self.current = self.max_point + self.current - self.min_point
            else:  # loop_mode == 'once'
                return self.finish()
        elif self.current >= self.max_point:
            if self.loop_mode == 2:  # 'loop_back_and_forth'
                self.step *= -1
                self.current = (
                    self.max_point + 2 * self.step * self.dimsrange[2]
                )
            elif self.loop_mode == 1:  # 'loop'
                self.current = self.min_point + self.current - self.max_point
            else:  # loop_mode == 'once'
                return self.finish()
        with self.dims.events.axis.blocker(self._on_axis_changed):
            self.frame_requested.emit(self.axis, self.current)
        self.timer.singleShot(self.interval, self.advance)

    def finish(self):
        self.finished.emit()

    @Slot(Event)
    def _on_axis_changed(self, event):
        # slot for external events to update the current frame
        if event.axis == self.axis and hasattr(event, 'value'):
            self.current = event.value


class QtDimSliderWidget(QWidget):
    """Compound widget to hold the label, slider and play button for an axis.

    These will usually be instantiated in the QtDims._create_sliders method.
    This widget *must* be instantiated with a parent QtDims.
    """

    label_changed = Signal(int, str)  # axis, label
    fps_changed = Signal(int)
    mode_changed = Signal(int)
    range_changed = Signal(tuple)
    play_started = Signal()
    play_stopped = Signal()

    def __init__(self, parent: QWidget, axis: int):
        super().__init__(parent=parent)
        self.axis = axis
        self.qt_dims = parent
        self.dims = parent.dims
        self.label = None
        self.slider = None
        self.play_button = None

        self._fps = 10
        self._minframe = None
        self._maxframe = None
        self._loop_mode = 1

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
        # rbutton = QtPlayButton(self.qt_dims, self.axis, True)
        self.play_button = QtPlayButton(self.qt_dims, self.axis)
        self.play_button.mode_combo.currentIndexChanged.connect(
            lambda x: self.__class__.loop_mode.fset(self, x)
        )

        def fps_listener():
            fps = self.play_button.fpsspin.value()
            self.__class__.fps.fset(self, fps)

        # I really don't like the way this works... I would rather connect
        # the listener to editing changed... but then it ignores the spinbox
        # buttons.  I tried for a while to fix that but couldn't well ... so
        # TODO: link listener only to editingFinished and button.clicked
        self.play_button.fpsspin.valueChanged.connect(fps_listener)
        self.play_stopped.connect(self.play_button._handle_stop)
        self.play_started.connect(self.play_button._handle_start)

    def _pull_label(self, event):
        if event.axis == self.axis:
            label = self.dims.axis_labels[self.axis]
            self.label.setText(label)
            self.label_changed.emit(self.axis, label)

    def _update_label(self):
        with self.dims.events.axis_labels.blocker():
            self.dims.set_axis_label(self.axis, self.label.text())
        self.label.clearFocus()
        self.qt_dims.setFocus()
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
            self.qt_dims.last_used = self.axis

        # linking focus listener to the last used:
        slider.sliderPressed.connect(slider_focused_listener)
        self.slider = slider

    def _update_range(self):
        """Updates range for slider."""
        displayed_sliders = self.qt_dims._displayed_sliders

        _range = self.dims.range[self.axis]
        _range = (_range[0], _range[1] - _range[2], _range[2])
        if _range not in (None, (None, None, None)):
            if _range[1] == 0:
                displayed_sliders[self.axis] = False
                self.qt_dims.last_used = None
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

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value
        self.play_button.fpsspin.setValue(value)
        self.fps_changed.emit(value)

    @property
    def loop_mode(self):
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, value):
        self._loop_mode = value
        self.play_button.mode_combo.setCurrentIndex(value)
        self.mode_changed.emit(value)

    @property
    def frame_range(self):
        frame_range = (self._minframe, self._maxframe)
        frame_range = frame_range if any(frame_range) else None
        return frame_range

    @frame_range.setter
    def frame_range(self, value):
        if not isinstance(value, (tuple, list, type(None))):
            raise TypeError('frame_range value must be a list or tuple')
        if value and not len(value) == 2:
            raise ValueError('frame_range must have a length of 2')
        self._minframe, self._maxframe = value
        self.range_changed.emit(tuple(value))

    def _update_play_settings(self, fps, frame_range, loop_mode):
        if fps is not None:
            self.fps = fps
        if frame_range is not None:
            self.frame_range = frame_range
        if loop_mode is not None:
            self.loop_mode = loop_mode

    def _play(
        self,
        fps: Optional[float] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        loop_mode: Optional[str] = None,
    ):
        """Animate (play) axis. """
        self._update_play_settings(fps, frame_range, loop_mode)

        # setting fps to 0 just stops the animation
        if fps == 0:
            return

        worker, thread = new_worker_qthread(
            AnimationWorker,
            self,
            start=True,
            connections={'frame_requested': self.qt_dims._set_frame},
        )
        thread.finished.connect(self.play_stopped.emit)
        self.play_started.emit()
        self.worker = worker
        self.thread = thread
        return worker, thread


class QtPlayButton(QPushButton):
    """Play button, included in the DimSliderWidget, to control playback

    the button also owns the QtModalPopup that controls the playback settings.
    """

    play_requested = Signal(int)  # axis, fps

    def __init__(self, dims, axis, reverse=False, fps=10, mode=1):
        super().__init__()
        self.dims = dims
        self.axis = axis
        self.reverse = reverse
        self.fps = fps
        self.mode = mode
        self.setProperty('reverse', str(reverse))  # for styling
        self.setProperty('playing', 'False')  # for styling

        # build popup modal form
        self.popup = QtModalPopup(self)
        self.fpsspin = QSpinBox(self.popup)
        self.fpsspin.setAlignment(Qt.AlignCenter)
        self.fpsspin.setValue(self.fps)
        self.fpsspin.setMaximum(500)
        self.fpsspin.setMinimum(-500)
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
        self.mode_combo.addItems(['play once', 'loop', 'back and forth'])
        self.popup.form_layout.insertRow(
            1, QLabel('play mode:', parent=self.popup), self.mode_combo
        )
        self.mode_combo.setCurrentIndex(self.mode)
        self.clicked.connect(self._on_click)

    def mouseReleaseEvent(self, event):
        # using this instead of self.customContextMenuRequested.connect
        # because the latter was not sending the rightMouseButton
        # release event.
        if event.button() == Qt.RightButton:
            self.popup.show_above_mouse()
        elif event.button() == Qt.LeftButton:
            self._on_click()

    def _on_click(self):
        if self.property('playing') == "True":
            return self.dims.stop()
        # TODO: link this to DimSliderWidget not QtDims
        self.play_requested.emit(self.axis)

    def _handle_start(self):
        self.setProperty('playing', 'True')
        self.style().unpolish(self)
        self.style().polish(self)

    def _handle_stop(self):
        self.setProperty('playing', 'False')
        self.style().unpolish(self)
        self.style().polish(self)
