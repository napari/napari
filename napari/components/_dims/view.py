from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QSlider, QGridLayout
from typing import Union

from napari._qt.range_slider.range_slider import QVRangeSlider, QHRangeSlider
from napari.components import Dims
from napari.components._dims.model import DimsMode, DimsEvent


class QtDims(QWidget):

    _slider_height = 22

    # Qt Signals for sending events to Qt thread
    update_axis = pyqtSignal(int)
    update_nbdims = pyqtSignal()

    # list of sliders
    sliders = []

    def __init__(self, dims: Dims):
        super().__init__()

        self.dims = dims


        self.setMinimumWidth(512)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        #self.setFixedHeight(0)

        self.num_sliders = dims.nb_dimensions

        # The next lines connect events coming from the model to the Qt event system:
        # We need to go through Qt signals so that these events are run in the Qt event loop
        # thread. This is all about changing thread context for thread-safety purposes

        # axis change listener
        def update_axis(source, axis):
            self.update_axis.emit(axis)
        self.dims.add_listener(DimsEvent.AxisChange, update_axis)

        # What to do with the axis change events in terms of UI calls to the widget
        self.update_axis.connect(self.update_slider)

        # nb dims change listener
        def update_nbdim(source):
            self.update_nbdims.emit()
        self.dims.add_listener(DimsEvent.NbDimChange, update_nbdim)

        # What to do with the nb dims change events in terms of UI calls to the widget
        self.update_nbdims.connect(self.update_nb_sliders)

    @property
    def num_sliders(self):
        return len(self.sliders)

    @num_sliders.setter
    def num_sliders(self, new_number_of_sliders):
        if self.num_sliders<new_number_of_sliders:
            self._create_sliders(new_number_of_sliders)
        elif self.num_sliders>new_number_of_sliders:
            self._trim_sliders(new_number_of_sliders)


    def update_slider(self, axis: int):

        if axis>=self.num_sliders:
            return

        slider = self.sliders[axis]

        if slider is None:
            return

        if axis<self.dims.nb_dimensions:
            if self.dims.get_mode(axis)==DimsMode.Point:
                slider.collapsed = True
                slider.setValue(self.dims.get_point(axis))
            elif self.dims.get_mode(axis)==DimsMode.Interval:
                slider.collapsed = False
                slider.setValues(self.dims.get_interval(axis))

    def update_nb_sliders(self):
        self.num_sliders = self.dims.nb_dimensions

    def _create_sliders(self, number_of_sliders):

        while number_of_sliders>self.num_sliders:
            new_slider_axis = self.num_sliders
            #slider = self._create_slider_widget()

            slider = self._create_range_slider_widget(new_slider_axis)
            #slider = self._create_slider_widget(new_slider_axis)
            self.layout().addWidget(slider, new_slider_axis, 0)
            self.sliders.append(slider)
            self.setFixedHeight(self.num_sliders * self._slider_height)

    def _trim_sliders(self, number_of_sliders):

        while number_of_sliders < self.num_sliders:
            slider = self.sliders.pop()
            self.layout().removeWidget(slider)
            slider.deleteLater()

    def _create_slider_widget(self, axis):
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setMinimum(0)
        slider.setFixedHeight(17)
        slider.setTickPosition(QSlider.NoTicks)
        # slider.setTickPosition(QSlider.TicksBothSides)
        # tick_interval = int(max(8,max_axis_length/8))
        # slider.setTickInterval(tick_interval)
        slider.setSingleStep(1)

        def value_change_listener(value):
            self.dims.set_mode(axis, DimsMode.Point)
            self.dims.set_point(axis, value)

        slider.valueChanged.connect(value_change_listener)

        return slider

    def _create_range_slider_widget(self, axis):

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


    #
    # class QtDims(QWidget):
    #     SLIDERHEIGHT = 19
    #
    #     def __init__(self, dims):
    #         super().__init__()
    #
    #         dims.events.update_slider.connect(self.update_slider)
    #         self._slider_value_changed = dims._slider_value_changed
    #         self.sliders = []
    #
    #         layout = QGridLayout()
    #         layout.setContentsMargins(0, 0, 0, 0)
    #         self.setLayout(layout)
    #         self.setFixedHeight(0)
    #
    #     def _axis_to_row(self, axis, max_dims):
    #         message = f'axis {axis} out of bounds for {max_dims} dims'
    #
    #         if axis < 0:
    #             axis = max_dims - axis
    #             if axis < 0:
    #                 raise IndexError(message)
    #         elif axis >= max_dims:
    #             raise IndexError(message)
    #
    #         if axis < 2:
    #             raise ValueError('cannot convert y/x-axes to rows')
    #
    #         return axis - 2
    #
    #     def update_slider(self, event):
    #         """Updates a slider for the given axis or creates
    #         it if it does not already exist.
    #
    #         Parameters
    #         ----------
    #         axis : int
    #             Axis that this slider controls.
    #         max_axis_length : int
    #             Longest length for this axis. If 0, deletes the slider.
    #
    #         Returns
    #         -------
    #         slider : PyQt5.QSlider or None
    #             Updated slider, if it exists.
    #         """
    #         axis = event.dim
    #         max_axis_length = event.dim_len
    #         max_dims = event.max_dims
    #
    #         grid = self.layout()
    #         row = self._axis_to_row(axis, max_dims)
    #
    #         slider = grid.itemAt(row)
    #         if max_axis_length <= 0:
    #             # delete slider
    #             grid.takeAt(row)
    #             return
    #
    #         if slider is None:  # has not been created yet
    #             # create slider
    #             if axis < 0:
    #                 raise ValueError('cannot create a slider '
    #                                  f'at negative axis {axis}')
    #
    #             slider = QSlider(Qt.Horizontal)
    #             slider.setFocusPolicy(Qt.StrongFocus)
    #             slider.setMinimum(0)
    #             slider.setFixedHeight(17)
    #             slider.setTickPosition(QSlider.NoTicks)
    #             # slider.setTickPosition(QSlider.TicksBothSides)
    #             # tick_interval = int(max(8,max_axis_length/8))
    #             # slider.setTickInterval(tick_interval)
    #             slider.setSingleStep(1)
    #
    #             grid.addWidget(slider, row, 0)
    #             self.sliders.append(slider)
    #         else:
    #             slider = slider.widget()
    #
    #         slider.valueChanged.connect(lambda value:
    #                                     self._slider_value_changed(value, axis))
    #         slider.setMaximum(max_axis_length - 1)
    #         self.setFixedHeight((max_dims-2)*self.SLIDERHEIGHT)
    #         return slider

