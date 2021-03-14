import logging
import pickle
from typing import Any, Generic, List, Optional, TypeVar

from qtpy.QtCore import QAbstractListModel, QMimeData, QModelIndex, Qt
from qtpy.QtWidgets import QWidget

from ....utils.events import disconnect_events
from ....utils.events.containers import SelectableEventedList

logger = logging.getLogger(__name__)
ItemType = TypeVar("ItemType")
ListIndexMIMEType = "application/x-list-index"


class QtListModel(QAbstractListModel, Generic[ItemType]):
    _list: SelectableEventedList[ItemType]

    def __init__(
        self,
        root: SelectableEventedList[ItemType],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setRoot(root)

    def rowCount(self, parent: QModelIndex = None) -> int:
        """Returns the number of rows under the given parent.

        When the parent is valid it means that rowCount is returning the number
        of children of parent.
        """
        return len(self._list)

    def data(self, index: QModelIndex, role: int) -> Any:
        """Return data stored under ``role`` for the item at ``index``.

        A given class:`QModelIndex` can store multiple types of data, each
        with its own "ItemDataRole".
        """
        if role == Qt.DisplayRole:
            return str(self.getItem(index))
        if role == Qt.UserRole:
            return self.getItem(index)
        return None

    def headerData(self, section, orientation, role) -> Any:
        # item-type specific subclasses may reimplement
        return super().headerData(section, orientation, role=role)

    def setData(self, index: QModelIndex, value: Any, role: int) -> bool:
        # item-type specific subclasses should reimplement
        return super().setData(index, value, role=role)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        # editable models must return a value containing Qt::ItemIsEditable.
        if (
            not index.isValid()
            or index.row() >= len(self._list)
            or index.model() is not self
        ):
            # we allow drops outside the items
            return Qt.ItemIsDropEnabled

        base_flags = (
            Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsEnabled
            | Qt.ItemNeverHasChildren
        )
        return base_flags

    def getItem(self, index: QModelIndex) -> ItemType:
        """Return item for a given `QModelIndex`.

        A null or invalid ``QModelIndex`` will return the root Node.
        """
        if index.isValid():
            return self._list[index.row()]

    def mimeTypes(self) -> List[str]:
        return [ListIndexMIMEType, "text/plain"]

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        destRow: int,
        col: int,
        parent: QModelIndex,
    ) -> bool:
        """Handles `data` from a drag and drop operation ending with `action`.

        The specified row, column and parent indicate the location of an item
        in the model where the operation ended. It is the responsibility of the
        model to complete the action at the correct location.

        Returns
        -------
        bool ``True`` if the ``data`` and ``action`` were handled by the model;
            otherwise returns ``False``.
        """
        if not data or action != Qt.MoveAction:
            return False
        if not data.hasFormat(self.mimeTypes()[0]):
            return False

        if isinstance(data, ItemMimeData):
            moving_indices = data.indices

            logger.debug(f"dropMimeData: indices {moving_indices} ➡ {destRow}")

            if len(moving_indices) == 1:
                self._list.move(moving_indices[0], destRow)
            else:
                self._list.move_multiple(moving_indices, destRow)
            return True
        return False

    def mimeData(self, indices: List[QModelIndex]) -> Optional['QMimeData']:
        """Return an object containing serialized data from `indices`."""
        # If the list of indexes is empty, or there are no supported MIME types,
        # nullptr is returned rather than a serialized empty list.
        if not indices:
            return 0
        items, indices = zip(*[(self.getItem(i), i.row()) for i in indices])
        return ItemMimeData(items, indices)

    def supportedDropActions(self) -> Qt.DropActions:
        """Returns the drop actions supported by this model."""
        return Qt.MoveAction

    # ##########################

    def setRoot(self, root: SelectableEventedList[ItemType]):
        if not isinstance(root, SelectableEventedList):
            raise TypeError(
                f"root node must be an instance of {SelectableEventedList}"
            )
        current_list = getattr(self, "_list", None)
        if root is current_list:
            return
        elif current_list is not None:
            disconnect_events(self._list.events, self)

        self._list = root
        self._list.events.removing.connect(self._on_begin_removing)
        self._list.events.removed.connect(self._on_end_remove)
        self._list.events.inserting.connect(self._on_begin_inserting)
        self._list.events.inserted.connect(self._on_end_insert)
        self._list.events.moving.connect(self._on_begin_moving)
        self._list.events.moved.connect(self._on_end_move)
        self._list.events.connect(self._process_event)

    def _process_event(self, event):
        # for subclasses to handle ItemType-specific data
        pass

    def _on_begin_removing(self, event):
        """Begins a row removal operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginRemoveRows
        """
        self.beginRemoveRows(QModelIndex(), event.index, event.index)

    def _on_end_remove(self, e):
        self.endRemoveRows()

    def _on_begin_inserting(self, event):
        """Begins a row insertion operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginInsertRows
        """
        self.beginInsertRows(QModelIndex(), event.index, event.index)

    def _on_end_insert(self, e):
        self.endInsertRows()

    def _on_begin_moving(self, event):
        """Begins a row move operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
        """
        src_par, src_idx = QModelIndex(), event.index
        dest_par, dest_idx = QModelIndex(), event.new_index

        logger.debug(f"beginMoveRows, {src_idx}->{dest_idx})")
        self.beginMoveRows(src_par, src_idx, src_idx, dest_par, dest_idx)

    def _on_end_move(self, e):
        self.endMoveRows()

    def findIndex(self, obj: ItemType) -> QModelIndex:
        """Find the QModelIndex for a given object in the model."""
        hits = self.match(
            self.index(0),
            Qt.UserRole,
            obj,
            1,
            Qt.MatchExactly | Qt.MatchRecursive,
        )
        if hits:
            return hits[0]
        return QModelIndex()


class ItemMimeData(QMimeData):
    def __init__(self, items, indices):
        super().__init__()
        self.items = items
        self.indices = indices
        if items:
            self.setData(ListIndexMIMEType, pickle.dumps(self.indices))
            self.setText(" ".join([str(item) for item in items]))

    def formats(self) -> List[str]:
        return [ListIndexMIMEType, "text/plain"]
