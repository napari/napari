from typing import Any, Iterable, Union

from qtpy.QtCore import QModelIndex, Qt

from napari._qt.containers.qt_list_model import QtListModel
from napari.components import Dims
from napari.utils.events import SelectableEventedList


class AxisModel:
    """View of an axis within a dims model.

    The model keeps track of axis names and allows read / write
    acces on the cossesponding rollable state of a Dims object.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Parent Dims object.
    axis : int
        Axis index.

    Attributes
    ----------
    dims : napari.components.dims.Dims
        Dimensions object modeling slicing and displaying.
    axis : int
        Axis index.
    """

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
        if not isinstance(other, AxisModel):
            return NotImplemented
        if isinstance(other, int):
            return self.axis == other
        return repr(self) == other

    @property
    def rollable(self) -> bool:
        """Get or set the 'rollable' state.

        Parameters
        ----------
        value : bool
            True if axis should be rollable.

        Returns
        -------
        bool
            True if axis is rollable.
        """
        return self.dims.rollable[self.axis]

    @rollable.setter
    def rollable(self, value: bool) -> None:
        rollable = list(self.dims.rollable)
        rollable[self.axis] = value
        self.dims.rollable = tuple(rollable)


class AxisList(SelectableEventedList[AxisModel]):
    def __init__(self, axes: Iterable[AxisModel]):
        super().__init__(axes)

    @classmethod
    def from_dims(cls, dims: Dims) -> 'AxisList':
        """Create AxisList instance from Dims object.

        The AxisList is filled with a number of AxisModels based
        on the number of dimensions in the Dims object.

        Parameters
        ----------
        dims : napari.components.dims.Dims
            Dims object to be used for creation.

        Returns
        -------
        AxisList
            A selectable evented list of the viewer axes.
        """
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
                if axis.rollable
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
            axis.rollable = bool(
                value == Qt.CheckState.Checked or value is True
            )
        elif role == Qt.ItemDataRole.EditRole:
            axis_labels = list(axis.dims.axis_labels)
            axis_labels[axis.axis] = value
            axis.dims.axis_labels = tuple(axis_labels)
        else:
            return super().setData(index, value, role=role)
        self.dataChanged.emit(index, index, [role])
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Returns the item flags for the given `index`.

        This describes the properties of a given item in the model.  We set
        them to be editable, checkable, draggable, droppable, etc...
        If index is not a list, we additionally set `Qt.ItemNeverHasChildren`
        (for optimization). Editable models must return a value containing
        `Qt.ItemIsEditable`.

        See Qt.ItemFlags https://doc.qt.io/qt-5/qt.html#ItemFlag-enum

        Parameters
        ----------
        index : qtpy.QtCore.QModelIndex
            Index to return flags for.

        Returns
        -------
        qtpy.QtCore.Qt.ItemFlags
            ItemFlags specific to the given index.
        """
        flags = (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemNeverHasChildren
        )

        if not index.isValid():
            # we allow drops outside the items
            return Qt.ItemFlag.ItemIsDropEnabled

        if self.getItem(index).rollable:
            # we only allow dragging if the item is rollable
            return flags | Qt.ItemFlag.ItemIsDragEnabled
        return flags
