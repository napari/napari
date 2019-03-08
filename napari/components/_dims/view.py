from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QSlider, QGridLayout
from typing import Union

from napari._qt.range_slider.range_slider import QVRangeSlider, QHRangeSlider
from napari.components import Dims
from napari.components._dims.model import DimsMode


class QtDims(QWidget):
    """
        Qt View for Dims model.
    """

    _slider_height = 22

    # Qt Signals for sending events to Qt thread
    update_axis = pyqtSignal(int)
    update_nbdims = pyqtSignal()

    # list of sliders
    sliders = []

    def __init__(self, dims: Dims):
        """
        Constructor for Dims View
        Parameters
        ----------
        dims : dims object
        """
        super().__init__()

        self.dims = dims


        self.setMinimumWidth(512)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        #self.setFixedHeight(0)

        self._set_num_sliders(dims.num_dimensions)

        # The next lines connect events coming from the model to the Qt event system:
        # We need to go through Qt signals so that these events are run in the Qt event loop
        # thread. This is all about changing thread context for thread-safety purposes

        # axis change listener
        def update_axis(event):
            self.update_axis.emit(event.axis)
        self.dims.changed.axis.connect(update_axis)

        # What to do with the axis change events in terms of UI calls to the widget
        self.update_axis.connect(self._update_slider)

        # nb dims change listener
        def update_nbdim(event):
            self.update_nbdims.emit()
        self.dims.changed.nbdims.connect(update_nbdim)

        # What to do with the nb dims change events in terms of UI calls to the widget
        self.update_nbdims.connect(self._update_nb_sliders)

    @property
    def num_sliders(self):
        """
        Returns the number of sliders displayed
        Returns
        -------
        output: number of sliders displayed
        """
        return len(self.sliders)

    def _update_slider(self, index: int):
        """
        Updates everything for a given slider
        Parameters
        ----------
        index : slider index (corresponds to axis index)

        """
        if index>=self.num_sliders:
            return

        slider = self.sliders[index]

        if slider is None:
            return

        if index<self.dims.num_dimensions:
            if self.dims.get_mode(index)==DimsMode.Point:
                slider.collapsed = True
                slider.setValue(self.dims.get_point(index))
            elif self.dims.get_mode(index)==DimsMode.Interval:
                slider.collapsed = False
                slider.setValues(self.dims.get_interval(index))

    def _update_nb_sliders(self):
        """

        """
        self._set_num_sliders(self.dims.num_dimensions)

    def _set_num_sliders(self, new_number_of_sliders):
        """
        Sets the number of sliders displayed
        Parameters
        ----------
        new_number_of_sliders :
        """
        if self.num_sliders < new_number_of_sliders:
            self._create_sliders(new_number_of_sliders)
        elif self.num_sliders > new_number_of_sliders:
            self._trim_sliders(new_number_of_sliders)

    def _create_sliders(self, number_of_sliders):
        """
        Creates sliders to match new number of dimensions
        Parameters
        ----------
        number_of_sliders : new number of sliders
        """
        while number_of_sliders>self.num_sliders:
            new_slider_axis = self.num_sliders
            #slider = self._create_slider_widget()

            slider = self._create_range_slider_widget(new_slider_axis)
            #slider = self._create_slider_widget(new_slider_axis)
            self.layout().addWidget(slider, new_slider_axis, 0)
            self.sliders.append(slider)
            self.setFixedHeight(self.num_sliders * self._slider_height)

    def _trim_sliders(self, number_of_sliders):
        """
        Trims number of dimensions to a lower number
        Parameters
        ----------
        number_of_sliders : new number of sliders
        """
        while number_of_sliders < self.num_sliders:
            slider = self.sliders.pop()
            self.layout().removeWidget(slider)
            slider.deleteLater()

    def _create_range_slider_widget(self, axis):
        """
        Creates a range slider widget for a given axis
        Parameters
        ----------
        axis : axis index

        Returns
        -------
        output : range slider
        """
        range = self.dims.get_range(axis)
        interval = self.dims.get_interval(axis)

        if range is None or range == (None, None, None):
            range = (0,100,1)
        if interval is None or interval == (None, None):
            interval = (0,100)


        slider = QHRangeSlider(slider_range=range,
                               values=interval,
                               parent=self)

        slider.default_collapse_logic=False
        slider.setFocusPolicy(Qt.StrongFocus)

        # notify of changes while sliding:
        slider.setEmitWhileMoving(True)

        # allows range slider to collapse to a single knob:
        slider.collapsable = True

        # and sets it in the correct state:
        slider.collapsed = self.dims.get_mode(axis) == DimsMode.Point

        # Listener to be used for sending events back to model:
        def slider_change_listener(min, max):
            if slider.collapsed:
                self.dims.set_point(axis, min)
            elif not slider.collapsed:
                self.dims.set_interval(axis, (min, max))

        # linking the listener to the slider:
        slider.rangeChanged.connect(slider_change_listener)

        # Listener to be used for sending events back to model:
        def collapse_change_listener(collapsed):
            if collapsed:
                interval = self.dims.get_interval(axis)
                if interval is not None:
                    min, max = interval
                    self.dims.set_point(axis, (max+min)/2)
            self.dims.set_mode(axis, DimsMode.Point if collapsed else DimsMode.Interval)

        slider.collapsedChanged.connect(collapse_change_listener)

        return slider


