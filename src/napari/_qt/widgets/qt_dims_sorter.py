from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

from ..._qt.containers import QtListView
from ...components import Dims
from ...utils.events import SelectableEventedList
from ...utils.translations import trans
from .qt_tooltip import QtToolTipLabel

if TYPE_CHECKING:
    from ...viewer import Viewer


class AxisModel:
    """View of an axis within a dims model keeping track of axis names."""

    def __init__(self, dims: Dims, axis: int):
        self.dims = dims
        self.axis = axis

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.dims.axis_labels[self.axis]

    def __eq__(self, other: Union[int, str]) -> bool:
        if isinstance(other, int):
            return self.axis == other
        else:
            return repr(self) == other


def set_dims_order(dims: Dims, order: Tuple[int, ...]):
    if type(order[0]) == AxisModel:
        order = [a.axis for a in order]
    dims.order = order


def _array_in_range(arr: np.ndarray, low: int, high: int) -> bool:
    return (arr >= low) & (arr < high)


def move_indices(axes_list: SelectableEventedList, order: Tuple[int, ...]):
    with axes_list.events.blocker_all():
        if tuple(axes_list) == tuple(order):
            return
        axes = [a.axis for a in axes_list]
        ax_to_existing_position = {a: ix for ix, a in enumerate(axes)}
        move_list = np.asarray(
            [(ax_to_existing_position[order[i]], i) for i in range(len(order))]
        )
        for src, dst in move_list:
            axes_list.move(src, dst)
            move_list[_array_in_range(move_list[:, 0], dst, src)] += 1
        # remove the elements from the back if order has changed length
        while len(axes_list) > len(order):
            axes_list.pop()


class QtDimsSorter(QWidget):
    """
    Modified from:
    https://github.com/jni/zarpaint/blob/main/zarpaint/_dims_chooser.py
    """

    def __init__(self, viewer: 'Viewer', parent=None):
        super().__init__(parent=parent)
        dims = viewer.dims
        root = SelectableEventedList(
            [AxisModel(dims, dims.order[i]) for i in range(dims.ndim)]
        )
        root.events.reordered.connect(
            lambda event: set_dims_order(dims, event.value)
        )
        dims.events.order.connect(
            lambda event: move_indices(root, event.value)
        )
        view = QtListView(root)
        view.setSizeAdjustPolicy(QtListView.AdjustToContents)

        self.axes_list = root

        layout = QGridLayout()
        self.setLayout(layout)

        widget_tooltip = QtToolTipLabel(self)
        widget_tooltip.setObjectName('help_label')
        widget_tooltip.setToolTip(trans._('Drag dimensions to reorder.'))

        widget_title = QLabel(trans._('Dims. Ordering'), self)

        self.layout().addWidget(widget_title, 0, 0)
        self.layout().addWidget(widget_tooltip, 0, 1)
        self.layout().addWidget(view, 1, 0)
