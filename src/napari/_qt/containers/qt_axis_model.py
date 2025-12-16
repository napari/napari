from collections.abc import Iterable
from typing import Any

from qtpy.QtCore import QModelIndex, Qt
from typing_extensions import Self

from napari._qt.containers.qt_list_model import QtListModel
from napari.components import Dims
from napari.utils.events import SelectableEventedList


class AxisModel:
    """View of an axis within a dims model.

    The model keeps track of axis names and allows read / write
    access on the corresponding rollable state of a Dims object.

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

    def __eq__(self, other: object) -> bool:
        # to allow comparisons between a list of AxisModels and the current dims order
        # we need to overload the int and str equality check, this is necessary as the
        # comparison will either be against a list of ints (Dims.order) or a list of
        # strings (Dims.axis_labels)
        if isinstance(other, int):
            return self.axis == other
        if isinstance(other, str):
            return repr(self) == other
        if isinstance(other, AxisModel):
            return (self.dims is other.dims) and (self.axis == other.axis)
        return NotImplemented

    @property
    def rollable(self) -> bool:
        """
        If the axis should be rollable.
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
    def from_dims(cls, dims: Dims) -> Self:
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
        return cls(AxisModel(dims, d) for d in dims.order)


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
            axis.rollable = Qt.CheckState(value) == Qt.CheckState.Checked
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
        if not index.isValid():
            # We only allow drops outside and in between the items
            # (and not inside them), in which case the index is not valid.
            return Qt.ItemFlag.ItemIsDropEnabled

        flags = (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemNeverHasChildren
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        return flags
