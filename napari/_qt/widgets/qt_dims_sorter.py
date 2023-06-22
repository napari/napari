from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

from napari._qt.containers import QtListView
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.components import Dims
from napari.settings import get_settings
from napari.utils.events import SelectableEventedList
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import Viewer


class AxisModel:
    """View of an axis within a dims model keeping track of axis names."""

    def __init__(self, dims: Dims, axis: int) -> None:
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

        return repr(self) == other


def set_dims_order(dims: Dims, order: Tuple[int, ...]):
    dims_order = list(dims.order)
    if type(order[0]) == AxisModel:
        order = [a.axis for a in order]
    dims_order[-dims.n_moveable_dims:] = order  # only modify the allowed number of dimensions
    dims.order = tuple(dims_order)


def _array_in_range(arr: np.ndarray, low: int, high: int) -> bool:
    return (arr >= low) & (arr < high)


def move_indices(axes_list: SelectableEventedList, order: Tuple[int, ...]):
    with axes_list.events.blocker_all():
        if tuple(axes_list) == tuple(order):
            return
        axes = [a.axis for a in axes_list]
        # only handle the allowed number of dimensions
        n_moveable_dims = axes_list[0].dims.n_moveable_dims
        order = order[-n_moveable_dims:]

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

    def __init__(self, viewer: 'Viewer', parent=None) -> None:
        super().__init__(parent=parent)
        self.dims = viewer.dims
        self.axes_list= SelectableEventedList(
            [AxisModel(self.dims, self.dims.order[-i]) for i in range(1, self.dims.n_moveable_dims + 1)][::-1]
        )
        self.axes_list.events.reordered.connect(
            lambda event: set_dims_order(self.dims, event.value)
        )
        self.dims.events.order.connect(
            lambda event: move_indices(self.axes_list, event.value)
        )
        # BUG: If one switches to n_movable_dims 2 rolls and goes back to more n_movable_dims rolling will still be just through two dims despite showing more in the QtDimsSorter, reordering in the widget resolves the bug. I probably have to connect an additional event somewhere.
        get_settings().application.events.n_moveable_dims.connect(
            # TODO: I would love to connect it to dims.events.n_movable_dims instead and use the same approach than the other two connections, but this causes update issues
            self._update_axes_list
        )
        view = QtListView(self.axes_list)
        view.setSizeAdjustPolicy(QtListView.AdjustToContents)

        layout = QGridLayout()
        self.setLayout(layout)

        widget_tooltip = QtToolTipLabel(self)
        widget_tooltip.setObjectName('help_label')
        widget_tooltip.setToolTip(trans._('Drag dimensions to reorder.'))

        widget_title = QLabel(trans._('Dims. Ordering'), self)

        self.layout().addWidget(widget_title, 0, 0)
        self.layout().addWidget(widget_tooltip, 0, 1)
        self.layout().addWidget(view, 1, 0)
    
    def _update_axes_list(self, event):
        if len(self.axes_list) != event.value:
            self.axes_list.clear()
            self.axes_list.extend(
                [AxisModel(self.dims, self.dims.order[-i]) for i in range(1, self.dims.n_moveable_dims + 1)][::-1]
            )
