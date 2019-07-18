from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget, QGridLayout, QSizePolicy

from . import QHRangeSlider
from ..components.dims import Dims
from ..components.dims_constants import DimsMode


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

    SLIDERHEIGHT = 26

    # Qt Signals for sending events to Qt thread
    update_axis = Signal(int)
    update_ndim = Signal()

    def __init__(self, dims: Dims, parent=None):

        super().__init__(parent=parent)

        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.sliders = []
        self._slider_axis = []

        # Initialises the layout:
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Update the number of sliders now that the dims have been added
        self._update_nsliders()

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

        # ndim change listener
        def update_ndim_listener(event):
            self.update_ndim.emit()

        self.dims.events.ndim.connect(update_ndim_listener)

        # What to do with the ndim change events in terms of UI calls to the
        # widget
        self.update_ndim.connect(self._update_nsliders)

    @property
    def nsliders(self):
        """Returns the number of sliders displayed

        Returns
        -------
        nsliders: int
            Number of sliders displayed
        """
        return len(self.sliders)

    def _update_slider(self, axis: int):
        """
        Updates everything for a given slider.

        Parameters
        ----------
        axis : int
            Axis index.
        """

        if axis not in self._slider_axis:
            return
        slider_index = self._slider_axis.index(axis)

        slider = self.sliders[slider_index]

        mode = self.dims.mode[axis]
        if mode == DimsMode.POINT:
            slider.collapse()
            slider.setValue(self.dims.point[axis])
        elif mode == DimsMode.INTERVAL:
            slider.expand()
            slider.setValues(self.dims.interval[axis])
        # Set the maximum values of the range slider to be one step less than
        # the range of the layer as otherwise the slider can move beyond the
        # shape of the layer as the endpoint is included
        range = self.dims.range[axis]
        range = (range[0], range[1] - range[2], range[2])
        if range not in (None, (None, None, None)):
            if range[1] == 0:
                self._remove_slider(slider_index)
            else:
                slider.setRange(range)
        else:
            self._remove_slider(slider_index)

        if self.dims.display[axis]:
            self._remove_slider(slider_index)

    def _update_nsliders(self):
        """
        Updates the number of sliders based on the number of dimensions
        """
        self._set_nsliders(self.dims.ndim - 2)
        self._slider_axis = list(range(self.dims.ndim - 2))
        for i in list(range(self.dims.ndim - 2)):
            self._update_slider(i)

    def _set_nsliders(self, new_number_of_sliders):
        """
        Sets the number of sliders displayed

        Parameters
        ----------
        new_number_of_sliders :
        """
        if self.nsliders < new_number_of_sliders:
            self._create_sliders(new_number_of_sliders)
        elif self.nsliders > new_number_of_sliders:
            self._trim_sliders(new_number_of_sliders)

    def _create_sliders(self, number_of_sliders):
        """
        Creates sliders to match new number of dimensions

        Parameters
        ----------
        number_of_sliders : new number of sliders
        """
        # add extra sliders so that number_of_sliders are present
        # add to the beginning of the list
        for slider_num in range(self.nsliders, number_of_sliders):
            dim_axis = number_of_sliders - slider_num - 1
            slider = self._create_range_slider_widget(dim_axis)
            self.layout().addWidget(slider)
            self.sliders.insert(0, slider)
            self._slider_axis = [a + 1 for a in self._slider_axis]
            self._slider_axis.insert(0, 0)
            self.setMinimumHeight(self.nsliders * self.SLIDERHEIGHT)

    def _trim_sliders(self, number_of_sliders):
        """
        Trims number of dimensions to a lower number

        Parameters
        ----------
        number_of_sliders : new number of sliders
        """
        # remove extra sliders so that only number_of_sliders are left
        # remove from the beginning of the list
        for slider_num in range(number_of_sliders, self.nsliders):
            self._remove_slider(0)

    def _remove_slider(self, index):
        """
        Remove slider at index

        Parameters
        ----------
        axis : int
            Index of slider to remove
        """
        # remove particular slider
        slider = self.sliders.pop(index)
        self._slider_axis.pop(index)
        self.layout().removeWidget(slider)
        slider.deleteLater()
        self.setMinimumHeight(self.nsliders * self.SLIDERHEIGHT)

    def _create_range_slider_widget(self, axis):
        """
        Creates a range slider widget for a given axis

        Parameters
        ----------
        axis : axis index

        Returns
        -------
        slider : range slider
        """
        range = self.dims.range[axis]
        # Set the maximum values of the range slider to be one step less than
        # the range of the layer as otherwise the slider can move beyond the
        # shape of the layer as the endpoint is included
        range = (range[0], range[1] - range[2], range[2])
        point = self.dims.point[axis]

        slider = QHRangeSlider(
            slider_range=range, values=(point, point), parent=self
        )

        slider.setFocusPolicy(Qt.StrongFocus)

        # notify of changes while sliding:
        slider.setEmitWhileMoving(True)

        # allows range slider to collapse to a single knob:
        slider.collapsable = True

        # and sets it in the correct state:
        if self.dims.mode[axis] == DimsMode.POINT:
            slider.collapse()
        else:
            slider.expand()

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
                interval = self.dims.interval[axis]
                if interval is not None:
                    min, max = interval
                    self.dims.set_point(axis, (max + min) / 2)
            self.dims.set_mode(
                axis, DimsMode.POINT if collapsed else DimsMode.INTERVAL
            )

        slider.collapsedChanged.connect(collapse_change_listener)

        return slider
