import warnings
from typing import Optional, Tuple

import numpy as np
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import QLineEdit, QSizePolicy, QVBoxLayout, QWidget

from ...components.dims import Dims
from ...settings._constants import LoopMode
from ...utils.translations import trans
from .qt_dims_slider import QtDimSliderWidget


class QtDims(QWidget):
    """Qt view for the napari Dims model.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Dims object to be passed to Qt object.
    parent : QWidget, optional
        QWidget that will be the parent of this widget.

    Attributes
    ----------
    dims : napari.components.dims.Dims
        Dimensions object modeling slicing and displaying.
    slider_widgets : list[QtDimSliderWidget]
        List of slider widgets.
    """

    def __init__(self, dims: Dims, parent=None):

        super().__init__(parent=parent)

        self.SLIDERHEIGHT = 22

        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.slider_widgets = []

        # True / False if slider is or is not displayed
        self._displayed_sliders = []

        self._play_ready = True  # False if currently awaiting a draw event
        self._animation_thread = None

        # Initialises the layout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Update the number of sliders now that the dims have been added
        self._update_nsliders()
        self.dims.events.ndim.connect(self._update_nsliders)
        self.dims.events.current_step.connect(self._update_slider)
        self.dims.events.range.connect(self._update_range)
        self.dims.events.ndisplay.connect(self._update_display)
        self.dims.events.order.connect(self._update_display)
        self.dims.events.last_used.connect(self._on_last_used_changed)

    @property
    def nsliders(self):
        """Returns the number of sliders.

        Returns
        -------
        nsliders: int
            Number of sliders.
        """
        return len(self.slider_widgets)

    def _on_last_used_changed(self):
        """Sets the style of the last used slider."""
        for i, widget in enumerate(self.slider_widgets):
            sld = widget.slider
            sld.setProperty('last_used', i == self.dims.last_used)
            sld.style().unpolish(sld)
            sld.style().polish(sld)

    def _update_slider(self):
        """Updates position for a given slider."""
        for widget in self.slider_widgets:
            widget._update_slider()

    def _update_range(self):
        """Updates range for a given slider."""
        for widget in self.slider_widgets:
            widget._update_range()

        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_slice_labels()

    def _update_display(self):
        """Updates display for all sliders."""
        self.stop()
        widgets = reversed(list(enumerate(self.slider_widgets)))
        nsteps = self.dims.nsteps
        for (axis, widget) in widgets:
            if axis in self.dims.displayed or nsteps[axis] <= 1:
                # Displayed dimensions correspond to non displayed sliders
                self._displayed_sliders[axis] = False
                self.dims.last_used = 0
                widget.hide()
            else:
                # Non displayed dimensions correspond to displayed sliders
                self._displayed_sliders[axis] = True
                self.dims.last_used = axis
                widget.show()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_slice_labels()

    def _update_nsliders(self):
        """Updates the number of sliders based on the number of dimensions."""
        self.stop()
        self._trim_sliders(0)
        self._create_sliders(self.dims.ndim)
        self._update_display()
        for i in range(self.dims.ndim):
            self._update_range()
            if self._displayed_sliders[i]:
                self._update_slider()

    def _resize_axis_labels(self):
        """When any of the labels get updated, this method updates all label
        widths to the width of the longest label. This keeps the sliders
        left-aligned and allows the full label to be visible at all times,
        with minimal space, without setting stretch on the layout.
        """
        fm = QFontMetrics(QFont("", 0))
        labels = self.findChildren(QLineEdit, 'axis_label')
        newwidth = max(fm.boundingRect(lab.text()).width() for lab in labels)

        if any(self._displayed_sliders):
            # set maximum width to no more than 20% of slider width
            maxwidth = self.slider_widgets[0].width() * 0.2
            newwidth = min([newwidth, maxwidth])
        for labl in labels:
            labl.setFixedWidth(int(newwidth) + 10)

    def _resize_slice_labels(self):
        """When the size of any dimension changes, we want to resize all of the
        slice labels to width of the longest label, to keep all the sliders
        right aligned.  The width is determined by the number of digits in the
        largest dimensions, plus a little padding.
        """
        width = 0
        for ax, maxi in enumerate(self.dims.nsteps):
            if self._displayed_sliders[ax]:
                length = len(str(maxi - 1))
                if length > width:
                    width = length
        # gui width of a string of length `width`
        fm = QFontMetrics(QFont("", 0))
        width = fm.boundingRect("8" * width).width()
        for labl in self.findChildren(QWidget, 'slice_label'):
            labl.setFixedWidth(width + 6)

    def _create_sliders(self, number_of_sliders: int):
        """Creates sliders to match new number of dimensions.

        Parameters
        ----------
        number_of_sliders : int
            New number of sliders.
        """
        # add extra sliders so that number_of_sliders are present
        # add to the beginning of the list
        for slider_num in range(self.nsliders, number_of_sliders):
            dim_axis = number_of_sliders - slider_num - 1
            slider_widget = QtDimSliderWidget(self, dim_axis)
            slider_widget.axis_label_changed.connect(self._resize_axis_labels)
            slider_widget.play_button.play_requested.connect(self.play)
            self.layout().addWidget(slider_widget)
            self.slider_widgets.insert(0, slider_widget)
            self._displayed_sliders.insert(0, True)
            nsliders = np.sum(self._displayed_sliders)
            self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_axis_labels()

    def _trim_sliders(self, number_of_sliders):
        """Trims number of dimensions to a lower number.

        Parameters
        ----------
        number_of_sliders : int
            New number of sliders.
        """
        # remove extra sliders so that only number_of_sliders are left
        # remove from the beginning of the list
        for slider_num in range(number_of_sliders, self.nsliders):
            self._remove_slider_widget(0)

    def _remove_slider_widget(self, index):
        """Remove slider_widget at index, including all sub-widgets.

        Parameters
        ----------
        index : int
            Index of slider to remove
        """
        # remove particular slider
        slider_widget = self.slider_widgets.pop(index)
        self._displayed_sliders.pop(index)
        self.layout().removeWidget(slider_widget)
        # As we delete this widget later, callbacks with a weak reference
        # to it may successfully grab the instance, but may be incompatible
        # with other update state like dims.
        self.dims.events.axis_labels.disconnect(slider_widget._pull_label)
        slider_widget.deleteLater()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(int(nsliders * self.SLIDERHEIGHT))
        self.dims.last_used = 0

    def play(
        self,
        axis: int = 0,
        fps: Optional[float] = None,
        loop_mode: Optional[str] = None,
        frame_range: Optional[Tuple[int, int]] = None,
    ):
        """Animate (play) axis.

        Parameters
        ----------
        axis : int
            Index of axis to play
        fps : float
            Frames per second for playback.  Negative values will play in
            reverse.  fps == 0 will stop the animation. The view is not
            guaranteed to keep up with the requested fps, and may drop frames
            at higher fps.
        loop_mode : str
            Mode for animation playback.  Must be one of the following options:
                "once": Animation will stop once movie reaches the
                    max frame (if fps > 0) or the first frame (if fps < 0).
                "loop":  Movie will return to the first frame
                    after reaching the last frame, looping until stopped.
                "back_and_forth":  Movie will loop back and forth until
                    stopped
        frame_range : tuple | list
            If specified, will constrain animation to loop [first, last] frames

        Raises
        ------
        IndexError
            If ``axis`` requested is out of the range of the dims
        IndexError
            If ``frame_range`` is provided and out of the range of the dims
        ValueError
            If ``frame_range`` is provided and range[0] >= range[1]
        """
        # doing manual check here to avoid issue in StringEnum
        # see https://github.com/napari/napari/issues/754
        if loop_mode is not None:
            _modes = LoopMode.keys()
            if loop_mode not in _modes:
                raise ValueError(
                    trans._(
                        'loop_mode must be one of {_modes}. Got: {loop_mode}',
                        _modes=_modes,
                        loop_mode=loop_mode,
                    )
                )
            loop_mode = LoopMode(loop_mode)

        if axis >= self.dims.ndim:
            raise IndexError(trans._('axis argument out of range'))

        if self.is_playing:
            if self._animation_worker.axis == axis:
                self.slider_widgets[axis]._update_play_settings(
                    fps, loop_mode, frame_range
                )
                return
            else:
                self.stop()

        # we want to avoid playing a dimension that does not have a slider
        # (like X or Y, or a third dimension in volume view.)
        if self._displayed_sliders[axis]:
            work = self.slider_widgets[axis]._play(fps, loop_mode, frame_range)
            if work:
                self._animation_worker, self._animation_thread = work
            else:
                self._animation_worker, self._animation_thread = None, None
        else:
            warnings.warn(
                trans._(
                    'Refusing to play a hidden axis',
                    deferred=True,
                )
            )

    def stop(self):
        """Stop axis animation"""
        if self._animation_thread:
            self._animation_thread.quit()
            self._animation_thread.wait()
        self._animation_thread = None
        self._animation_worker = None
        self.enable_play()

    @property
    def is_playing(self):
        """Return True if any axis is currently animated."""
        return self._animation_thread and self._animation_thread.isRunning()

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
            self.dims.set_current_step(axis, frame)

    def enable_play(self, *args):
        # this is mostly here to connect to the main SceneCanvas.events.draw
        # event in the qt_viewer
        self._play_ready = True

    def closeEvent(self, event):
        [w.deleteLater() for w in self.slider_widgets]
        self.deleteLater()
        event.accept()
