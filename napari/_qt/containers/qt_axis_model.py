from typing import Any, Iterable, Union

from qtpy.QtCore import QModelIndex, Qt

from napari._qt.containers.qt_list_model import QtListModel
from napari.components import Dims
from napari.utils.events import SelectableEventedList


class AxisModel:
    """View of an axis within a dims model keeping track of axis names."""

    def __init__(self, dims: Dims, axis: int) -> None:
        self.dims = dims
        self.axis = axis

    def rollable(self) -> bool:
        return self.dims.rollable[self.axis]

    def set_rollable(self, r: bool) -> None:
        rollable = list(self.dims.rollable)
        rollable[self.axis] = r
        self.dims.rollable = rollable

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


class AxisList(SelectableEventedList[AxisModel]):
    def __init__(self, axes: Iterable[AxisModel]):
        super().__init__(axes)

    @classmethod
    def from_dims(cls, dims: Dims) -> 'AxisList':
        return cls([AxisModel(dims, dims.order[i]) for i in range(dims.ndim)])


class QtAxisListModel(QtListModel[AxisModel]):
    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        if not index.isValid():
            return None
        axis = self.getItem(index)
        if role == Qt.ItemDataRole.DisplayRole:
            return str(axis)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignCenter
        if role == Qt.ItemDataRole.CheckStateRole:
            return (
                Qt.CheckState.Checked
                if axis.rollable()
                else Qt.CheckState.Unchecked
            )
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        axis = self.getItem(index)
        if role == Qt.ItemDataRole.CheckStateRole:
            axis.set_rollable(Qt.CheckState(value) == Qt.CheckState.Checked)
        else:
            return super().setData(index, value, role=role)
        self.dataChanged.emit(index, index, [role])
        return True

    def _process_event(self, event):
        # The model needs to emit `dataChanged` whenever data has changed
        # for a given index, so that views can update themselves.
        # Here we convert native events to the dataChanged signal.
        if not hasattr(event, 'index'):
            return
        role = {
            'name': Qt.ItemDataRole.DisplayRole,
            'rollable': Qt.ItemDataRole.CheckStateRole,
        }.get(event.type)
        roles = [role] if role is not None else []
        row = self.index(event.index)
        self.dataChanged.emit(row, row, roles)
