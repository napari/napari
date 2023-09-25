from typing import Tuple

import numpy as np
from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

from napari._qt.containers import QtListView
from napari._qt.containers.qt_axis_model import AxisList, AxisModel
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.components import Dims
from napari.utils.events import SelectableEventedList
from napari.utils.translations import trans


def set_dims_order(dims: Dims, order: Tuple[int, ...]):
    """Set dimension order of Dims object to order.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Dims object.
    order : tuple of int
        New dimension order.
    """
    if type(order[0]) == AxisModel:
        order = [a.axis for a in order]
    dims.order = order


def _array_in_range(arr: np.ndarray, low: int, high: int) -> bool:
    return (arr >= low) & (arr < high)


# TODO: This function is in the current usecase of QtDimsSorter unnecessary and could be removed
def move_indices(axis_list: SelectableEventedList, order: Tuple[int, ...]):
    with axis_list.events.blocker_all():
        if tuple(axis_list) == tuple(order):
            return
        axes = [a.axis for a in axis_list]
        ax_to_existing_position = {a: ix for ix, a in enumerate(axes)}
        move_list = np.asarray(
            [(ax_to_existing_position[order[i]], i) for i in range(len(order))]
        )
        for src, dst in move_list:
            axis_list.move(src, dst)
            move_list[_array_in_range(move_list[:, 0], dst, src)] += 1
        # remove the elements from the back if order has changed length
        while len(axis_list) > len(order):
            axis_list.pop()


class QtDimsSorter(QWidget):
    """Qt widget for dimension / axis reordering and locking.

    Modified from:
    https://github.com/jni/zarpaint/blob/main/zarpaint/_dims_chooser.py

    Parameters
    ----------
    viewer : napari.Viewer
        Main napari viewer instance.
    parent : QWidget
        QWidget that holds this widget. A QtDimsSorter instances will
        disconnect all callbacks upon closing of it's parent.

    Attributes
    ----------
    dims : napari.components.Dims
        Dimensions object of the current viewer, modeling slicing and displaying.
    axis_list : napari._qt.containers.qt_axis_model.AxisList
        Selectable evented list representing the viewer axes.
    """

    def __init__(self, dims: Dims, parent: QWidget) -> None:
        super().__init__(parent=parent)
        self.dims = dims
        self.axis_list = AxisList.from_dims(self.dims)

        view = QtListView(self.axis_list)
        if len(self.axis_list) <= 2:
            view.setSizeAdjustPolicy(QtListView.AdjustToContents)

        layout = QGridLayout()
        self.setLayout(layout)

        widget_tooltip = QtToolTipLabel(self)
        widget_tooltip.setObjectName('help_label')
        widget_tooltip.setToolTip(
            trans._(
                'Drag dimensions to reorder, click lock icon to lock dimension in place.'
            )
        )

        widget_title = QLabel(trans._('Dims. Ordering'), self)

        self.layout().addWidget(widget_title, 0, 0)
        self.layout().addWidget(widget_tooltip, 0, 1)
        self.layout().addWidget(view, 1, 0, 1, 2)

        # connect axis_list and dims
        # terminate connections after parent widget is closed
        # to allow closure of QtDimsSorter
        self.axis_list.events.reordered.connect(
            self._axis_list_reorder_callback,
            until=self.parent().finished,
        )
        self.parent().destroyed.connect(lambda: print('parent dead'))
        # self.dims.events.order.connect(
        #    self._dims_order_callback,
        #    until=self.parent().destroyed,
        # )

    def _axis_list_reorder_callback(self, event):
        set_dims_order(self.dims, event.value)

    def _dims_order_callback(self, event):
        move_indices(self.axis_list, event.value)
