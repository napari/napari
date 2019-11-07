from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QScrollBar,
    QLineEdit,
)
import numpy as np

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

    # Qt Signals for sending events to Qt thread
    update_ndim = Signal()
    update_axis = Signal(int)
    update_range = Signal(int)
    update_display = Signal()
    update_axis_labels = Signal(int)

    def __init__(self, dims: Dims, parent=None):

        super().__init__(parent=parent)

        self.SLIDERHEIGHT = 22

        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.sliders = []

        self.axis_labels = []
        # True / False if slider is or is not displayed
        self._displayed_sliders = []

        self._last_used = None

        # Initialises the layout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
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

        def update_axis_labels_listener(event):
            self.update_axis_labels.emit(event.axis_labels)

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
        """
        Updates position for a given slider.

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
        """
        Updates range for a given slider.

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
        for (axis, slider), label in zip(slider_list, label_list):
            if axis in self.dims.displayed:
                # Displayed dimensions correspond to non displayed sliders
                self._displayed_sliders[axis] = False
                self.last_used = None
                slider.hide()
                label.hide()
            else:
                # Non displayed dimensions correspond to displayed sliders
                self._displayed_sliders[axis] = True
                self.last_used = axis
                slider.show()
                label.show()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)

    def _update_nsliders(self):
        """
        Updates the number of sliders based on the number of dimensions
        """
        self._trim_sliders(0)
        self._create_sliders(self.dims.ndim)
        self._update_display()
        for i in range(self.dims.ndim):
            self._update_range(i)
            if self._displayed_sliders[i]:
                self._update_slider(i)

    def _update_axis_labels(self, axis):
        """Updates the label for the given axis """
        # TODO
        self.dims.axis_labels[axis] = 'dsads'

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
            axis_label = self._create_axis_label_widget(dim_axis)
            slider = self._create_range_slider_widget(dim_axis)

            # Hard-coded 1:50 ratio. Can be more dynamic as a function
            # of the name of the label, but it might be a little bit
            # over the top.
            current_row = QHBoxLayout()
            if axis_label.text != '':
                current_row.addWidget(axis_label, stretch=1)
                current_row.addWidget(slider, stretch=50)
            else:
                current_row.addWidget(slider)
            self.layout().addLayout(current_row)
            self.axis_labels.insert(0, axis_label)
            self.sliders.insert(0, slider)
            self._displayed_sliders.insert(0, True)
            nsliders = np.sum(self._displayed_sliders)
            self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)

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
        Remove slider at index. Should also remove all accompanying widgets,
        like the axis label.

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
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self.last_used = None

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
        _range = self.dims.range[axis]
        # Set the maximum values of the range slider to be one step less than
        # the range of the layer as otherwise the slider can move beyond the
        # shape of the layer as the endpoint is included
        _range = (_range[0], _range[1] - _range[2], _range[2])
        point = self.dims.point[axis]

        slider = QScrollBar(Qt.Horizontal)
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
        """
        Create the axis label widget which accompanies its slider.
        # TODO - min and max width, word wrapping, text alignmemt.
        Parameters
        ----------
        axis : axis index

        Returns
        -------
        label : QLabel with the given text
        """
        label = QLineEdit()
        label.setText(self.dims.axis_labels[axis])
        label.home(False)
        label.setToolTip('Axis label')
        label.setAcceptDrops(False)
        label.setEnabled(True)
        # label.setFocusPolicy(Qt.NoFocus)

        def axis_label_change_listener(value):
            self.dims.axis_labels[axis] = value

        label.editingFinished.connect(axis_label_change_listener)
        return label

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
