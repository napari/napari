from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QSlider, QGridLayout, QRadioButton
from typing import Union

from napari._qt.range_slider.range_slider import QVRangeSlider, QHRangeSlider

from napari.components._dims.dims import DimsMode, Dims


class QtDims(QWidget):
    """Qt View for Dims model.

    Parameters
    ----------
    dims : Dims
        Dims object to be passed to Qt object
    parent : QWidget, optional
        QWidget that will be the parent of this widget
    """

    _slider_height = 22

    # Qt Signals for sending events to Qt thread
    update_axis = pyqtSignal(int)
    update_ndims = pyqtSignal()


    def __init__(self, dims: Dims, parent = None):

        super().__init__(parent=parent)


        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.sliders = []

        # adjust this widget initial geometry hints:
        self.setMinimumWidth(512)
        self.setMinimumHeight(self._slider_height)

        # Initialises the layout:
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        #layout.setColumnStretch(1, 1)
        self.setLayout(layout)


        # First we need to make sure that the current state of the model is
        # correctly reflected in the view. This is important because in the
        # general case, we might not have model and view synced if changes have
        # occured before the view is initialised with the model.

        # First we set the dimenions of the view with respect to the model:
        self._set_num_sliders(dims.num_dimensions)

        # Then we set the mode fop each slider:

        for axis in range(0, dims.num_dimensions):
            slider = self.sliders[axis]
            if self.dims.get_mode(axis)==DimsMode.Point:
                slider.collapse()
            else:
                slider.expand()


        # The next lines connect events coming from the model to the Qt event
        # system: We need to go through Qt signals so that these events are run
        # in the Qt event loop thread. This is all about changing thread
        # context for thread-safety purposes

        # axis change listener
        def update_axis_listener(event):
            self.update_axis.emit(event.axis)
        self.dims.events.axis.connect(update_axis_listener)

        # What to do with the axis change events in terms of UI calls to the
        # widget
        self.update_axis.connect(self._update_slider)

        # nb dims change listener
        def update_nbdim_listener(event):
            self.update_ndims.emit()
        self.dims.events.ndims.connect(update_nbdim_listener)

        # What to do with the nb dims change events in terms of UI calls to the
        # widget
        self.update_ndims.connect(self._update_nb_sliders)

    @property
    def num_sliders(self):
        """
        Returns the number of sliders displayed
        Returns
        -------
        output: number of sliders displayed
        """
        return len(self.sliders)

    def _update_slider(self, slider_index: int):
        """
        Updates everything for a given slider
        Parameters
        ----------
        slider_index : slider index (corresponds to axis index)

        """
        if slider_index>=self.num_sliders:
            return

        slider = self.sliders[slider_index]

        if slider is None:
            return

        if slider_index<self.dims.num_dimensions:

            mode = self.dims.get_mode(slider_index)
            if mode ==DimsMode.Point:
                slider.collapsed = True
                slider.setValue(self.dims.get_point(slider_index))
            elif mode ==DimsMode.Interval:
                slider.collapsed = False
                slider.setValues(self.dims.get_interval(slider_index))
            slider_range = self.dims.get_range(slider_index)
            if (slider_range is not None) and (slider_range != (None, None, None)):
                slider.setRange(slider_range)

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
            #self.layout().addWidget(QRadioButton(), new_slider_axis, 0)
            self.layout().addWidget(slider, new_slider_axis, 0)
            #self.layout().setRowMinimumHeight(new_slider_axis, self._slider_height)
            self.sliders.append(slider)
            self.setMinimumHeight(self.num_sliders * self._slider_height)

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

        #if range is None or range == (None, None, None):
        #    range = (0,100,1)
        #if interval is None or interval == (None, None):
        #    interval = (0,100)


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
