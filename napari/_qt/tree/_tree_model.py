from __future__ import annotations

import pickle
from typing import Any, List, Tuple, Union

from qtpy.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt
from qtpy.QtWidgets import QWidget
from ...utils.tree import Group, Node


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtNodeTreeModel(QAbstractItemModel):
    def __init__(self, root: Group, parent: QWidget = None):
        super().__init__(parent)
        self.root_item = root
        self.root_item.events.removing.connect(self._on_begin_removing)
        self.root_item.events.removed.connect(lambda x: self.endRemoveRows())
        self.root_item.events.inserting.connect(self._on_begin_inserting)
        self.root_item.events.inserted.connect(lambda x: self.endInsertRows())
        # self.root_item.events.added.connect(self._on_added)

    def _on_begin_removing(self, event):
        par, idx = QModelIndex(), event.value
        self.beginRemoveRows(par, idx, idx)

    def _on_begin_inserting(self, event):
        par, idx = QModelIndex(), event.value
        self.beginInsertRows(par, idx, idx)

    # def _on_added(self, event):
    #     for idx, _ in event.value:
    #         par, idx = self._split_nested_index(idx)
    #         self.beginInsertRows(par, idx, idx)
    #         self.endInsertRows()
    #     return True

    # def _split_nested_index(
    #     self, nested_index: Union[int, Tuple[int, ...]]
    # ) -> Tuple[QModelIndex, int]:
    #     """Given a nested index, return (nested_parent_index, row)."""
    #     if isinstance(nested_index, int):
    #         return QModelIndex(), nested_index
    #     par = QModelIndex()
    #     *_p, idx = nested_index
    #     for i in _p:
    #         par = self.index(i, 0, par)
    #     return par, idx

    def canDropMimeData(self, *args):
        return self.getItem(args[-1]).is_group()

    def columnCount(self, parent: QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        """Return data stored under ``role`` for the item at ``index``."""
        item = self.getItem(index)
        if role == Qt.DisplayRole:
            return str(item.name)
        return None

    def getItem(self, index: QModelIndex) -> Node:
        if index.isValid():
            item = index.internalPointer()
            if item is not None:
                return item
        return self.root_item

    def flags(self, index: QModelIndex) -> Union[Qt.ItemFlag, Qt.ItemFlags]:
        """Returns the item flags for the given index."""
        if not index.isValid():
            # for root
            return Qt.ItemIsDropEnabled

        base_flags = (
            Qt.ItemIsSelectable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsEnabled
            | Qt.ItemIsDropEnabled
        )
        item = self.getItem(index)
        if item.is_group():
            return base_flags
        return base_flags | Qt.ItemNeverHasChildren

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Return index of the item specified by row, column and parent index.
        """
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        parentItem = self.getItem(parent)
        if parentItem.is_group():
            return self.createIndex(row, column, parentItem[row])
        return QModelIndex()

    # def insertRows(self, pos: int, count: int, parent: QModelIndex) -> bool:
    #     parentItem = self.getItem(parent)
    #     if pos < 0 or pos > len(parentItem):
    #         return False

    #     self.beginInsertRows(parent, pos, pos + count - 1)
    #     for i in range(count):
    #         item = Node()
    #         parentItem.insert(pos, item)
    #     self.endInsertRows()

    #     return True

    def moveRows(
        self,
        sourceParent: QModelIndex,
        sourceRow: int,
        count: int,
        destinationParent: QModelIndex,
        destinationChild: int,
    ) -> bool:
        """moves count rows starting with the sourceRow under sourceParent
        to row destinationChild under destinationParent."""
        destParentItem = self.getItem(destinationParent)
        if destinationChild > len(destParentItem):
            return False
        if destinationChild < 0:
            destinationChild = len(destParentItem)

        srcParentItem = self.getItem(sourceParent)
        self.beginMoveRows(
            sourceParent,
            sourceRow,
            sourceRow + count - 1,
            destinationParent,
            destinationChild,
        )
        # same parent
        if srcParentItem == destParentItem:
            if destinationChild > sourceRow:
                destinationChild -= count
            if sourceRow == destinationChild:
                return False
        for i in range(count):
            with srcParentItem.events.removed.blocker():
                item = srcParentItem.pop(sourceRow)
            with destParentItem.events.added.blocker():
                destParentItem.insert(destinationChild, item)
        self.endMoveRows()

        return True

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()

        parentItem = self.getItem(index).parent

        if parentItem is None or parentItem == self.root_item:
            return QModelIndex()

        return self.createIndex(parentItem.index_in_parent(), 0, parentItem)

    # def removeRows(self, pos: int, count: int, parent: QModelIndex):
    #     print("RemoveRows")
    #     parentItem = self.getItem(parent)
    #     if pos < 0 or (pos + count) > len(parentItem):
    #         return False

    #     self.beginRemoveRows(parent, pos, pos + count - 1)
    #     for i in range(count):
    #         parentItem.pop(pos)
    #     self.endRemoveRows()

    #     return True

    def rowCount(self, parent: QModelIndex) -> int:
        return len(self.getItem(parent))

    def supportedDropActions(self) -> Qt.DropActions:
        return Qt.MoveAction

    def mimeTypes(self):
        return ['application/x-layertree', 'text/plain']

    def mimeData(self, indices: List[QModelIndex]) -> QMimeData:
        """Return object containing serialized data corresponding to indexes.
        """
        if not indices:
            return 0

        mimedata = QMimeData()
        data = []
        text = []
        for index in indices:
            item = self.getItem(index)
            data.append(item.index_from_root())
            text.append(item.name)
        mimedata.setData(self.mimeTypes()[0], pickle.dumps(data))
        mimedata.setText(" ".join(text))
        return mimedata

    # def dropMimeData(
    #     self,
    #     data: QMimeData,
    #     action: Qt.DropAction,
    #     destRow: int,
    #     col: int,
    #     parent: QModelIndex,
    # ) -> bool:
    #     """Handles dropped data that ended with ``action``.

    #     Returns true if the data and action were handled by the model;
    #     otherwise returns false.

    #     """
    #     if not data or action != Qt.MoveAction:
    #         return False
    #     default_format = self.mimeTypes()[0]
    #     if not data.hasFormat(default_format):
    #         return False

    #     dest_parent_item = self.getItem(parent)
    #     dest_tup = dest_parent_item.index_from_root()
    #     if destRow == -1:
    #         destRow = len(dest_parent_item)

    #     dragged_indices = pickle.loads(data.data(default_format))
    #     if len(dragged_indices) <= 1:
    #         # simpler task
    #         for *p, srcRow in dragged_indices:
    #             src_parent = self.get_nested_index(p)
    #             self.moveRows(src_parent, srcRow, 1, parent, destRow)
    #         return False

    #     # more complicated when moving multiple objects.
    #     # don't assume selection adjacency ... so move one at a time
    #     # need to update indices as we pop, so we keep track of the indices
    #     # we have previously popped
    #     popped: DefaultDict[Tuple[int, ...], List[int]] = defaultdict(list)
    #     # we iterate indices from the end first, so pop() always works
    #     for i, (*p, srcRow) in enumerate(
    #         sorted(dragged_indices, reverse=True)
    #     ):
    #         src_tup = tuple(p)  # index of parent relative to root
    #         src_parent = self.get_nested_index(src_tup)

    #         # we need to decrement the srcRow by 1 for each time we have
    #         # previously pulled items out from in front of the srcRow
    #         sdec = sum(map(lambda x: x <= srcRow, popped.get(src_tup, [])))
    #         # if item is being moved within the same parent,
    #         # we need to increase the srcRow by 1 for each time we have
    #         # previously inserted items in front of the srcRow
    #         if src_tup == dest_tup:
    #             sdec -= (destRow <= srcRow) * i
    #         # we need to decrement the destRow by 1 for each time we have
    #         # previously pulled items out from in front of the destRow
    #         ddec = sum(map(lambda x: x <= destRow, popped.get(dest_tup, [])))

    #         self.moveRows(src_parent, srcRow - sdec, 1, parent, destRow - ddec)
    #         popped[src_tup].append(srcRow)

    #     # If we return true, removeRows is called!?
    #     return False

    def get_nested_index(self, indices: Tuple[int, ...]) -> QModelIndex:
        parentIndex = QModelIndex()
        for idx in indices:
            parentIndex = self.index(idx, 0, parentIndex)
        return parentIndex

    # def iter_indices(self, parent: QModelIndex = QModelIndex()):
    #     if parent.isValid():
    #         yield parent
    #     for c in range(self.rowCount(parent)):
    #         yield from self.iter_indices(self.index(c, 0, parent))

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        destRow: int,
        col: int,
        parent: QModelIndex,
    ) -> bool:
        """Handles dropped data that ended with ``action``.

        Returns true if the data and action were handled by the model;
        otherwise returns false.

        """
        if not data or action != Qt.MoveAction:
            return False
        default_format = self.mimeTypes()[0]
        if not data.hasFormat(default_format):
            return False
        dragged_indices = pickle.loads(data.data(default_format))

        dest_idx = list(self.getItem(parent).index_from_root())
        dest_idx.append(destRow)

        self.root_item.move_multiple_nested(dragged_indices, tuple(dest_idx))

        # If we return true, removeRows is called!?
        return False


if __name__ == '__main__':
    from napari import gui_qt
    from ._tree_view import QtNodeTreeView

    with gui_qt():
        tip = Node(name='tip')
        lg2 = Group(name="g2", children=[Node(name='2')])
        lg1 = Group(
            name="g1",
            children=[
                lg2,
                Node(name='3'),
                tip,
                Node(name='1'),
                Node(name='4'),
                Node(name='5'),
            ],
        )
        root = Group(
            name="root",
            children=[
                lg1,
                Node(name='6'),
                Node(name='7'),
                Node(name='8'),
                Node(name='9'),
            ],
        )
        tree = QtNodeTreeView(root)
        model = tree.model()
        tree.show()
