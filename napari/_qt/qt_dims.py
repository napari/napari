from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import QEventLoop, QThread, QTimer, Signal, Slot
from qtpy.QtWidgets import QVBoxLayout, QSizePolicy, QWidget, QLineEdit

from ..components.dims import Dims
from ..components.dims_constants import DimsMode
from ..util.event import Event
from .qt_dims_slider import QtDimSliderWidget
from qtpy.QtGui import QFont, QFontMetrics


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
    play_started = Signal(int, int)
    play_stopped = Signal()

    def __init__(self, dims: Dims, parent=None):

        super().__init__(parent=parent)

        self.SLIDERHEIGHT = 22

        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.slider_widgets = []

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

    @property
    def nsliders(self):
        """Returns the number of sliders displayed

        Returns
        -------
        nsliders: int
            Number of sliders displayed
        """
        return len(self.slider_widgets)

    @property
    def last_used(self):
        """int: Index of slider last used.
        """
        return self._last_used

    @last_used.setter
    def last_used(self, last_used: int):
        if last_used == self.last_used:
            return

        formerly_used = self.last_used
        if formerly_used is not None:
            sld = self.slider_widgets[formerly_used].slider
            sld.setProperty('last_used', False)
            sld.style().unpolish(sld)
            sld.style().polish(sld)

        self._last_used = last_used
        if last_used is not None:
            sld = self.slider_widgets[last_used].slider
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

        if axis >= len(self.slider_widgets):
            return

        slider = self.slider_widgets[axis].slider

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
        if axis >= len(self.slider_widgets):
            return

        self.slider_widgets[axis]._update_range()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)

    def _update_display(self):
        """Updates display for all sliders."""
        widgets = reversed(list(enumerate(self.slider_widgets)))
        for (axis, widget) in widgets:
            if axis in self.dims.displayed:
                # Displayed dimensions correspond to non displayed sliders
                self._displayed_sliders[axis] = False
                self.last_used = None
                widget.hide()
            else:
                # Non displayed dimensions correspond to displayed sliders
                self._displayed_sliders[axis] = True
                self.last_used = axis
                widget.show()
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

    def _resize_labels(self):
        """When any of the labels get updated, this method updates all label
        widths to the width of the longest label. This keeps the sliders
        left-aligned and allows the full label to be visible at all times,
        with minimal space, without setting stretch on the layout.
        """
        fm = QFontMetrics(QFont("", 0))
        labels = self.findChildren(QLineEdit, 'axis_label')
        newwidth = max([fm.width(lab.text()) for lab in labels])

        if any(self._displayed_sliders):
            # set maximum width to no more than 20% of slider width
            maxwidth = self.slider_widgets[0].width() * 0.2
            newwidth = min([newwidth, maxwidth])
        for labl in labels:
            labl.setFixedWidth(newwidth + 10)

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
            slider_widget = QtDimSliderWidget(dim_axis, self)
            slider_widget.label_changed.connect(self._resize_labels)
            slider_widget.play_button.play_requested.connect(self.play)
            self.layout().addWidget(slider_widget)
            self.slider_widgets.insert(0, slider_widget)
            self._displayed_sliders.insert(0, True)
            nsliders = np.sum(self._displayed_sliders)
            self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_labels()

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
            self._remove_slider_widget(0)

    def _remove_slider_widget(self, index):
        """Remove slider_widget at index, including all sub-widgets.

        Parameters
        ----------
        axis : int
            Index of slider to remove
        """
        # remove particular slider
        slider_widget = self.slider_widgets.pop(index)
        self._displayed_sliders.pop(index)
        self.layout().removeWidget(slider_widget)
        slider_widget.deleteLater()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self.last_used = None

    def focus_up(self):
        """Shift focused dimension slider to be the next slider above."""
        displayed = list(np.nonzero(self._displayed_sliders)[0])
        if len(displayed) == 0:
            return

        if self.last_used is None:
            # this code may be unreachable?
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
            # this code may be unreachable?
            self.last_used = displayed[-1]
        else:
            index = (displayed.index(self.last_used) - 1) % len(displayed)
            self.last_used = displayed[index]

    def play(
        self,
        axis: int = 0,
        fps: Optional[float] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        playback_mode: Optional[str] = None,
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
        # allow only one axis to be playing at a time
        # if nothing is playing self.stop() will not do anything
        self.stop()
        if playback_mode:
            playback_mode = playback_mode.lower()
            _modes = {'loop', 'once', 'loop_back_and_forth'}
            if playback_mode not in _modes:
                raise ValueError(
                    f'"{playback_mode}" not a recognized playback_mode: ({_modes})'
                )

        # if the play() function was not called with parameters, we default to
        # the current values set in the GUI.  This allows the
        if fps is None:
            fps = self.slider_widgets[axis].play_button.fps
        else:
            self.slider_widgets[axis].play_button.set_fps(fps)
        if playback_mode is None:
            playback_mode = self.slider_widgets[axis].play_button.mode
        else:
            self.slider_widgets[axis].play_button.set_mode(playback_mode)

        if fps == 0:
            return

        # play() is a main front-facing api, so when play() is called
        # for a particular axis and fps, we store the values and update
        # listeners (so playbutton will remember the last fps)

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
        # self._animation_thread.finished.connect(
        #     lambda: self.play_stopped.emit()
        # )

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
        self.playback_mode = playback_mode

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
        # if playback_mode is once and we are already on the last frame,
        # return to the first frame... (so the user can keep hitting once)
        if self.playback_mode == 'once':
            if self.step > 0 and self.current >= self.max_point - 1:
                self.incremented.emit(self.axis, self.min_point)
            elif self.step < 0 and self.current <= self.min_point + 1:
                self.incremented.emit(self.axis, self.max_point)
        else:
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
                self.parent().stop()
                # self.quit()
        elif self.current >= self.max_point:
            if self.playback_mode == 'loop_back_and_forth':
                self.step *= -1
                self.current = (
                    self.max_point + 2 * self.step * self.dimsrange[2]
                )
            elif self.playback_mode == 'loop':
                self.current = self.min_point + self.current - self.max_point
            else:  # self.playback_mode == 'once'
                self.parent().stop()
        with self.dims.events.axis.blocker(self._on_axis_changed):
            self.incremented.emit(self.axis, self.current)

    @Slot(Event)
    def _on_axis_changed(self, event):
        # slot for external events to update the current frame
        if event.axis == self.axis and hasattr(event, 'value'):
            self.current = event.value
