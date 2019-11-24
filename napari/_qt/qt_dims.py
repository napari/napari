from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import QEventLoop, Qt, QThread, QTimer, Signal, Slot, QPoint
from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
    QLineEdit,
    QPushButton,
    QDialog,
    QFormLayout,
    QLabel,
    QFrame,
    QSpinBox,
)
from qtpy.QtGui import QCursor

from ..components.dims import Dims
from ..components.dims_constants import DimsMode
from ..util.event import Event
from .qt_scrollbar import ModifiedScrollBar


class QtDims(QWidget):
    """Qt View for Dims model.

    Parameters
    ----------
    dims : Dims
        Dims object to be passed to Qt object
    parent : QWidget, optional
        QWidget that will be the parent of this widget

    Attributes
    ----------
    dims : Dims
        Dims object
    sliders : list
        List of slider widgets
    """

    # Qt Signals for sending events to Qt thread
    update_ndim = Signal()
    update_axis = Signal(int)
    update_range = Signal(int)
    update_display = Signal()
    update_axis_labels = Signal(int)
    play_started = Signal(int, int)
    play_stopped = Signal()

    def __init__(self, dims: Dims, parent=None):

        super().__init__(parent=parent)

        self.SLIDERHEIGHT = 22

        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.sliders = []

        self.axis_labels = []
        self.play_buttons = []
        # True / False if slider is or is not displayed
        self._displayed_sliders = []

        self._last_used = None
        self._play_ready = True  # False if currently awaiting a draw event

        # Initialises the layout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Update the number of sliders now that the dims have been added
        self._update_nsliders()

        # The next lines connect events coming from the model to the Qt event
        # system: We need to go through Qt signals so that these events are run
        # in the Qt event loop thread. This is all about changing thread
        # context for thread-safety purposes

        # ndim change listener
        def update_ndim_listener(event):
            self.update_ndim.emit()

        self.dims.events.ndim.connect(update_ndim_listener)
        self.update_ndim.connect(self._update_nsliders)

        # axis change listener
        def update_axis_listener(event):
            self.update_axis.emit(event.axis)

        self.dims.events.axis.connect(update_axis_listener)
        self.update_axis.connect(self._update_slider)

        # range change listener
        def update_range_listener(event):
            self.update_range.emit(event.axis)

        self.dims.events.range.connect(update_range_listener)
        self.update_range.connect(self._update_range)

        # display change listener
        def update_display_listener(event):
            self.update_display.emit()

        self.dims.events.ndisplay.connect(update_display_listener)
        self.dims.events.order.connect(update_display_listener)
        self.update_display.connect(self._update_display)

        # axis labels change listener
        def update_axis_labels_listener(event):
            self.update_axis_labels.emit(event.axis)

        self.dims.events.axis_labels.connect(update_axis_labels_listener)
        self.update_axis_labels.connect(self._update_axis_labels)

    @property
    def nsliders(self):
        """Returns the number of sliders displayed

        Returns
        -------
        nsliders: int
            Number of sliders displayed
        """
        return len(self.sliders)

    @property
    def last_used(self):
        """int: Index of slider last used.
        """
        return self._last_used

    @last_used.setter
    def last_used(self, last_used):
        if last_used == self.last_used:
            return

        formerly_used = self.last_used
        if formerly_used is not None:
            sld = self.sliders[formerly_used]
            sld.setProperty('last_used', False)
            sld.style().unpolish(sld)
            sld.style().polish(sld)

        self._last_used = last_used
        if last_used is not None:
            sld = self.sliders[last_used]
            sld.setProperty('last_used', True)
            sld.style().unpolish(sld)
            sld.style().polish(sld)

    def _update_slider(self, axis: int):
        """Updates position for a given slider.

        Parameters
        ----------
        axis : int
            Axis index.
        """

        if axis >= len(self.sliders):
            return

        slider = self.sliders[axis]

        mode = self.dims.mode[axis]
        if mode == DimsMode.POINT:
            slider.setValue(self.dims.point[axis])
        self.last_used = axis

    def _update_range(self, axis: int):
        """Updates range for a given slider.

        Parameters
        ----------
        axis : int
            Axis index.
        """
        if axis >= len(self.sliders):
            return

        slider = self.sliders[axis]

        _range = self.dims.range[axis]
        _range = (_range[0], _range[1] - _range[2], _range[2])
        if _range not in (None, (None, None, None)):
            if _range[1] == 0:
                self._displayed_sliders[axis] = False
                self.last_used = None
                slider.hide()
            else:
                if (
                    not self._displayed_sliders[axis]
                    and axis not in self.dims.displayed
                ):
                    self._displayed_sliders[axis] = True
                    self.last_used = axis
                    slider.show()
                slider.setMinimum(_range[0])
                slider.setMaximum(_range[1])
                slider.setSingleStep(_range[2])
                slider.setPageStep(_range[2])
        else:
            self._displayed_sliders[axis] = False
            slider.hide()

        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)

    def _update_display(self):
        """Updates display for all sliders."""
        slider_list = reversed(list(enumerate(self.sliders)))
        label_list = reversed(self.axis_labels)
        play_buttons = reversed(self.play_buttons)
        for (axis, slider), label, (rplay, fplay) in zip(
            slider_list, label_list, play_buttons
        ):
            if axis in self.dims.displayed:
                # Displayed dimensions correspond to non displayed sliders
                self._displayed_sliders[axis] = False
                self.last_used = None
                slider.hide()
                label.hide()
                rplay.hide()
                fplay.hide()
            else:
                # Non displayed dimensions correspond to displayed sliders
                self._displayed_sliders[axis] = True
                self.last_used = axis
                slider.show()
                label.show()
                rplay.show()
                fplay.show()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)

    def _update_nsliders(self):
        """Updates the number of sliders based on the number of dimensions."""
        self._trim_sliders(0)
        self._create_sliders(self.dims.ndim)
        self._update_display()
        for i in range(self.dims.ndim):
            self._update_range(i)
            if self._displayed_sliders[i]:
                self._update_slider(i)

    def _update_axis_labels(self, axis):
        """Updates the label for the given axis."""
        self.axis_labels[axis].setText(self.dims.axis_labels[axis])

    def _create_sliders(self, number_of_sliders: int):
        """Creates sliders to match new number of dimensions.

        Parameters
        ----------
        number_of_sliders : int
            new number of sliders
        """
        # add extra sliders so that number_of_sliders are present
        # add to the beginning of the list
        for slider_num in range(self.nsliders, number_of_sliders):
            dim_axis = number_of_sliders - slider_num - 1
            axis_label = self._create_axis_label_widget(dim_axis)
            slider = self._create_range_slider_widget(dim_axis)
            rbutton, fbutton = self._create_play_button_widgets(dim_axis)
            # Hard-coded 1:50 ratio. Can be more dynamic as a function
            # of the name of the label, but it might be a little bit
            # over the top.
            current_row = QHBoxLayout()

            if axis_label.text != '':
                current_row.addWidget(axis_label, stretch=1.5)
                current_row.addWidget(rbutton, stretch=1)
                current_row.addWidget(slider, stretch=50)
                current_row.addWidget(fbutton, stretch=1)
            else:
                current_row.addWidget(rbutton, stretch=1)
                current_row.addWidget(slider, stretch=50)
                current_row.addWidget(fbutton, stretch=1)
            self.layout().addLayout(current_row)
            self.axis_labels.insert(0, axis_label)
            self.sliders.insert(0, slider)
            self.play_buttons.insert(0, (rbutton, fbutton))
            self._displayed_sliders.insert(0, True)
            nsliders = np.sum(self._displayed_sliders)
            self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)

    def _trim_sliders(self, number_of_sliders):
        """Trims number of dimensions to a lower number.

        Parameters
        ----------
        number_of_sliders : int
            new number of sliders
        """
        # remove extra sliders so that only number_of_sliders are left
        # remove from the beginning of the list
        for slider_num in range(number_of_sliders, self.nsliders):
            self._remove_slider(0)

    def _remove_slider(self, index):
        """Remove slider at index, including all accompanying widgets.

        Parameters
        ----------
        axis : int
            Index of slider to remove
        """
        # remove particular slider
        slider = self.sliders.pop(index)
        self._displayed_sliders.pop(index)
        self.layout().removeWidget(slider)
        axis_label = self.axis_labels.pop(index)
        self.layout().removeWidget(axis_label)
        slider.deleteLater()
        axis_label.deleteLater()
        for button in self.play_buttons.pop(index):
            self.layout().removeWidget(button)
            button.deleteLater()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self.last_used = None

    def _create_range_slider_widget(self, axis: int):
        """Creates a range slider widget for a given axis.

        Parameters
        ----------
        axis : int
            axis index

        Returns
        -------
            slider : range slider
        """
        _range = self.dims.range[axis]
        # Set the maximum values of the range slider to be one step less than
        # the range of the layer as otherwise the slider can move beyond the
        # shape of the layer as the endpoint is included
        _range = (_range[0], _range[1] - _range[2], _range[2])
        point = self.dims.point[axis]

        slider = ModifiedScrollBar(Qt.Horizontal)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(_range[0])
        slider.setMaximum(_range[1])
        slider.setSingleStep(_range[2])
        slider.setPageStep(_range[2])
        slider.setValue(point)

        # Listener to be used for sending events back to model:
        def slider_change_listener(value):
            self.dims.set_point(axis, value)

        # linking the listener to the slider:
        slider.valueChanged.connect(slider_change_listener)

        def slider_focused_listener():
            self.last_used = self.sliders.index(slider)

        # linking focus listener to the last used:
        slider.sliderPressed.connect(slider_focused_listener)

        return slider

    def _create_axis_label_widget(self, axis):
        """Create the axis label widget which accompanies its slider.

        Parameters
        ----------
        axis : int
            axis index

        Returns
        -------
        label : QLabel
            A label with the given text
        """
        label = QLineEdit()
        label.setText(self.dims.axis_labels[axis])
        label.home(False)
        label.setToolTip('Axis label')
        label.setAcceptDrops(False)
        label.setEnabled(True)

        def changeText():
            with self.dims.events.axis_labels.blocker():
                self.dims.set_axis_label(axis, label.text())
            label.clearFocus()
            self.setFocus()

        label.editingFinished.connect(changeText)
        return label

    def _create_play_button_widgets(self, axis):
        rbutton = QtPlayButton(self, axis, True)
        fbutton = QtPlayButton(self, axis)
        return rbutton, fbutton

    def focus_up(self):
        """Shift focused dimension slider to be the next slider above."""
        displayed = list(np.nonzero(self._displayed_sliders)[0])
        if len(displayed) == 0:
            return

        if self.last_used is None:
            self.last_used = displayed[-1]
        else:
            index = (displayed.index(self.last_used) + 1) % len(displayed)
            self.last_used = displayed[index]

    def focus_down(self):
        """Shift focused dimension slider to be the next slider bellow."""
        displayed = list(np.nonzero(self._displayed_sliders)[0])
        if len(displayed) == 0:
            return

        if self.last_used is None:
            self.last_used = displayed[-1]
        else:
            index = (displayed.index(self.last_used) - 1) % len(displayed)
            self.last_used = displayed[index]

    def play(
        self,
        axis: int = 0,
        fps: float = 10,
        frame_range: Optional[Tuple[int, int]] = None,
        playback_mode: str = 'loop',
    ):
        """Animate (play) axis.

        Parameters
        ----------
        axis: int
            Index of axis to play
        fps: float
            Frames per second for playback.  Negative values will play in
            reverse.  fps == 0 will stop the animation. The view is not
            guaranteed to keep up with the requested fps, and may drop frames
            at higher fps.
        frame_range: tuple | list
            If specified, will constrain animation to loop [first, last] frames
        playback_mode: str
            Mode for animation playback.  Must be one of the following options:
                'loop': Movie will return to the first frame after reaching
                    the last frame, looping until stopped.
                'once': Animation will stop once movie reaches the max frame
                    (if fps > 0) or the first frame (if fps < 0).
                'loop_back_and_forth':  Movie will loop back and forth until
                    stopped

        Raises
        ------
        IndexError
            If ``axis`` requested is out of the range of the dims
        IndexError
            If ``frame_range`` is provided and out of the range of the dims
        ValueError
            If ``frame_range`` is provided and range[0] >= range[1]
        """
        # TODO: No access in the GUI yet. Just keybinding.

        # allow only one axis to be playing at a time
        # if nothing is playing self.stop() will not do anything
        self.stop()
        if fps == 0:
            return

        if axis >= len(self.dims.range):
            raise IndexError('axis argument out of range')
        # we want to avoid playing a dimension that does not have a slider
        # (like X or Y, or a third dimension in volume view.)
        if not self._displayed_sliders[axis]:
            return

        self._animation_thread = AnimationThread(
            self.dims, axis, fps, frame_range, playback_mode, parent=self
        )
        # when the thread timer increments, update the current frame
        self._animation_thread.incremented.connect(self._set_frame)
        self._animation_thread.start()
        self.play_started.emit(axis, fps)

    def stop(self):
        """Stop axis animation"""
        if self.is_playing:
            self._animation_thread.quit()
            self._animation_thread.wait()
            del self._animation_thread
            self.enable_play()
            self.play_stopped.emit()

    @property
    def is_playing(self):
        """Return True if any axis is currently animated."""
        return (
            hasattr(self, '_animation_thread')
            # this is repetive, since we delete the thread each time, but safer
            and self._animation_thread.isRunning()
        )

    def _set_frame(self, axis, frame):
        """Safely tries to set `axis` to the requested `point`.

        This function is debounced: if the previous frame has not yet drawn to
        the canvas, it will simply do nothing.  If the timer plays faster than
        the canvas can draw, this will drop the intermediate frames, keeping
        the effective frame rate constant even if the canvas cannot keep up.
        """
        if self._play_ready:
            # disable additional point advance requests until this one draws
            self._play_ready = False
            self.dims.set_point(axis, frame)

    def enable_play(self, *args):
        # this is mostly here to connect to the main SceneCanvas.events.draw
        # event in the qt_viewer
        self._play_ready = True


class ModalPopup(QDialog):
    """A generic modal popup window.

    The seemingly extra frame here is to allow rounded corners on a truly
    transparent background

    +-------------------------------
    | Dialog
    |  +----------------------------
    |  | QVBoxLayout
    |  |  +-------------------------
    |  |  | QFrame
    |  |  |  +----------------------
    |  |  |  | QFormLayout
    |  |  |  |
    |  |  |  |  (stuff goes here)
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("QtModalPopup")
        self.setModal(True)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setLayout(QVBoxLayout())

        self.frame = QFrame()
        self.layout().addWidget(self.frame)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.form_layout = QFormLayout()
        closebutton = QPushButton("Close")
        closebutton.clicked.connect(self.close)
        self.form_layout.addRow(closebutton)
        self.frame.setLayout(self.form_layout)

    def show_above_mouse(self, *args):
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(szhint.width() / 2, szhint.height() + 14)
        self.move(pos)
        self.show()


class QtPlayButton(QPushButton):
    def __init__(self, dims, axis, reverse=False, fps=10):
        super().__init__()
        self.dims = dims
        self.axis = axis
        self.reverse = reverse
        self.fps = fps
        self.setProperty('reverse', str(reverse))
        self.setProperty('playing', 'False')
        self.clicked.connect(self._on_click)

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
        print(dimsrange)
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

    def _on_context_menu(self, point):
        pos = QCursor().pos()  # mouse position
        szhint = self.popup.sizeHint()
        pos -= QPoint(szhint.width() / 2, szhint.height() + 14)
        self.popup.move(pos)
        self.popup.show()

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
        print('clicked')
        if self.dims.is_playing:
            if self.dims._animation_thread.axis == self.axis:
                if (self.dims._animation_thread.step == -1) == self.reverse:
                    return self.dims.stop()
        self.dims.play(self.axis, self.fps * (-1 if self.reverse else 1))


class AnimationThread(QThread):
    """A thread to keep the animation timer independent of the main event loop.

    This prevents mouseovers and other events from causing animation lag. See
    QtDims.play() for public-facing docstring.
    """

    incremented = Signal(int, int)  # signal for each time a frame is requested

    def __init__(
        self,
        dims,
        axis,
        fps=10,
        frame_range=None,
        playback_mode='loop',
        parent=None,
    ):
        super().__init__(parent)
        # could put some limits on fps here... though the handler in the QtDims
        # object above is capable of ignoring overly spammy requests.

        _mode = playback_mode.lower()
        _modes = {'loop', 'once', 'loop_back_and_forth'}
        if _mode not in _modes:
            raise ValueError(
                f'"{_mode}" not a recognized playback_mode: ({_modes})'
            )

        self.dims = dims
        self.axis = axis

        self.dimsrange = self.dims.range[axis]
        if frame_range is not None:
            if frame_range[0] >= frame_range[1]:
                raise ValueError("frame_range[0] must be <= frame_range[1]")
            if frame_range[0] < self.dimsrange[0]:
                raise IndexError("frame_range[0] out of range")
            if frame_range[1] * self.dimsrange[2] >= self.dimsrange[1]:
                raise IndexError("frame_range[1] out of range")
        self.frame_range = frame_range
        self.playback_mode = _mode

        if self.frame_range is not None:
            self.min_point, self.max_point = self.frame_range
        else:
            self.min_point = 0
            self.max_point = int(
                np.floor(self.dimsrange[1] - self.dimsrange[2])
            )
        self.max_point += 1  # range is inclusive

        # after dims.set_point is called, it will emit a dims.events.axis()
        # we use this to update this threads current frame (in case it
        # was some other event that updated the axis)
        self.dims.events.axis.connect(self._on_axis_changed)
        self.current = max(self.dims.point[axis], self.min_point)
        self.current = min(self.current, self.max_point)
        self.step = 1 if fps > 0 else -1  # negative fps plays in reverse
        self.timer = QTimer()
        self.timer.setInterval(1000 / abs(fps))
        self.timer.timeout.connect(self.advance)
        self.timer.moveToThread(self)
        # this is necessary to avoid a warning in QtDims.stop() on del thread
        self.finished.connect(self.timer.deleteLater)

    def run(self):
        # immediately advance one frame
        self.advance()
        self.timer.start()
        loop = QEventLoop()
        loop.exec_()

    def advance(self):
        """Advance the current frame in the animation.

        Takes dims scale into account and restricts the animation to the
        requested frame_range, if entered.
        """
        self.current += self.step * self.dimsrange[2]
        if self.current < self.min_point:
            if self.playback_mode == 'loop_back_and_forth':
                self.step *= -1
                self.current = self.min_point + self.step * self.dimsrange[2]
            elif self.playback_mode == 'loop':
                self.current = self.max_point + self.current - self.min_point
            else:  # self.playback_mode == 'once'
                self.quit()
        elif self.current >= self.max_point:
            if self.playback_mode == 'loop_back_and_forth':
                self.step *= -1
                self.current = (
                    self.max_point + 2 * self.step * self.dimsrange[2]
                )
            elif self.playback_mode == 'loop':
                self.current = self.min_point + self.current - self.max_point
            else:  # self.playback_mode == 'once'
                self.quit()
        with self.dims.events.axis.blocker(self._on_axis_changed):
            self.incremented.emit(self.axis, self.current)

    @Slot(Event)
    def _on_axis_changed(self, event):
        # slot for external events to update the current frame
        if event.axis == self.axis and hasattr(event, 'value'):
            self.current = event.value
