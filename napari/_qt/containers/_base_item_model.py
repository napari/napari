from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, Tuple, TypeVar, Union

from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt

from ...utils.events import disconnect_events
from ...utils.events.containers import EventedList, SelectableEventedList

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


ItemType = TypeVar("ItemType")
logger = logging.getLogger(__name__)


class _BaseItemModel(QAbstractItemModel, Generic[ItemType]):
    def __init__(
        self, root: SelectableEventedList[ItemType], parent: QWidget = None
    ):
        super().__init__(parent=parent)
        self.setRoot(root)

    def setRoot(self, root: SelectableEventedList[ItemType]):
        if not isinstance(root, EventedList):
            raise TypeError(f"root node must be an instance of {EventedList}")
        current_root = getattr(self, "_root", None)
        if root is current_root:
            return
        elif current_root is not None:
            disconnect_events(self._root.events, self)

        self._root = root
        self._root.events.removing.connect(self._on_begin_removing)
        self._root.events.removed.connect(self._on_end_remove)
        self._root.events.inserting.connect(self._on_begin_inserting)
        self._root.events.inserted.connect(self._on_end_insert)
        self._root.events.moving.connect(self._on_begin_moving)
        self._root.events.moved.connect(self._on_end_move)
        self._root.events.connect(self._process_event)

    def data(self, index: QModelIndex, role: int) -> Any:
        """Return data stored under ``role`` for the item at ``index``.

        A given class:`QModelIndex` can store multiple types of data, each
        with its own "ItemDataRole".  ItemType-specific subclasses will likely
        want to customize this method for different data roles.
        """
        if role == Qt.DisplayRole:
            return str(self.getItem(index))
        if role == Qt.UserRole:
            return self.getItem(index)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Returns the item flags for the given index.

        Editable models must return a value containing Qt::ItemIsEditable.
        """
        if (
            not index.isValid()
            or index.row() >= len(self._root)
            or index.model() is not self
        ):
            # we allow drops outside the items
            return Qt.ItemIsDropEnabled

        return (
            Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsEnabled
            | Qt.ItemNeverHasChildren
        )

    def _split_nested_index(
        self, nested_index: Union[int, Tuple[int, ...]]
    ) -> Tuple[QModelIndex, int]:
        """Given a nested index, return (nested_parent_index, row)."""
        if isinstance(nested_index, int):
            return QModelIndex(), nested_index
        par = QModelIndex()
        *_p, idx = nested_index
        for i in _p:
            par = self.index(i, 0, par)
        return par, idx

    def _on_begin_removing(self, event):
        """Begins a row removal operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginRemoveRows
        """
        par, idx = self._split_nested_index(event.index)
        self.beginRemoveRows(par, idx, idx)

    def _on_begin_inserting(self, event):
        """Begins a row insertion operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginInsertRows
        """
        par, idx = self._split_nested_index(event.index)
        self.beginInsertRows(par, idx, idx)

    def _on_begin_moving(self, event):
        """Begins a row move operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
        """
        src_par, src_idx = self._split_nested_index(event.index)
        dest_par, dest_idx = self._split_nested_index(event.new_index)

        logger.debug(
            f"beginMoveRows({self.getItem(src_par)._node_name()}, {src_idx}, "
            f"{self.getItem(dest_par)._node_name()}, {dest_idx})"
        )

        self.beginMoveRows(src_par, src_idx, src_idx, dest_par, dest_idx)

    def _on_end_insert(self, e):
        self.endInsertRows()

    def _on_end_remove(self, e):
        self.endRemoveRows()

    def _on_end_move(self, e):
        self.endMoveRows()

    def columnCount(self, parent: QModelIndex) -> int:
        """Returns the number of columns for the children of the given ``parent``.

        In a tree and a list view, the number of columns is always 1.
        """
        return 1

    def findIndex(self, obj: ItemType) -> QModelIndex:
        """Find the QModelIndex for a given object in the model."""
        hits = self.match(
            self.index(0, 0),
            Qt.UserRole,
            obj,
            1,
            Qt.MatchExactly | Qt.MatchRecursive,
        )
        if hits:
            return hits[0]
        return QModelIndex()

    def _process_event(self, event):
        # for subclasses to handle ItemType-specific data
        pass

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Return index of the item specified by ``row``, ``column`` and ``parent`` index."""
        return (
            self.createIndex(row, column, self.getItem(parent)[row])
            if self.hasIndex(row, column, parent)
            else QModelIndex()
        )

    def getItem(self, index: QModelIndex) -> ItemType:
        """Return ``Node`` object for a given `QModelIndex`.

        A null or invalid ``QModelIndex`` will return the root Node.
        """
        if index.isValid():
            item = index.internalPointer()
            if item is not None:
                return item
        return self._root

    def rowCount(self, parent: QModelIndex) -> int:
        """Returns the number of rows under the given parent.

        When the parent is valid it means that rowCount is returning the number of
        children of parent.
        """
        try:
            return len(self.getItem(parent))
        except TypeError:
            return 0

    def supportedDropActions(self) -> Qt.DropActions:
        """Returns the drop actions supported by this model.

        The default implementation returns Qt::CopyAction. We re-implement to support only
        MoveAction. See also dropMimeData(), which must handles each operation.
        """
        return Qt.MoveAction
