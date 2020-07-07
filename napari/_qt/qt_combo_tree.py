from __future__ import annotations

import pickle
from abc import ABC
from collections import defaultdict
from typing import (
    Any,
    DefaultDict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from napari.utils.list._evented_list import NestableEventedList
from qtpy.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt
from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget


class Node(ABC):
    def __init__(self, name: str = 'Node'):
        self._parent: Optional[Group] = None
        self.name = name

    def __len__(self) -> int:
        return 0

    @property
    def parent(self) -> Optional[Group]:
        return self._parent

    @parent.setter
    def parent(self, parent: Group):
        self._parent = parent

    def is_group(self) -> bool:
        return False

    def index_in_parent(self) -> int:
        if self.parent is not None:
            return self.parent.index(self)
        return 0

    def index_from_root(self) -> Tuple[int, ...]:
        indices: List[int] = []
        item = self
        while item.parent is not None:
            indices.insert(0, item.index_in_parent())
            item = item.parent
        return tuple(indices)

    def traverse(self):
        yield self

    def __str__(self):
        """Render ascii tree string representation of this node"""
        return "\n".join(self._render())

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        return [self.name]


class Leaf(Node):
    pass


class Group(Node, NestableEventedList):
    def __init__(self, name='Group', children: Iterable[Node] = None) -> None:
        Node.__init__(self, name=name)
        NestableEventedList.__init__(self)
        self.extend(children or [])

    def __len__(self) -> int:
        return NestableEventedList.__len__(self)

    def __delitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            self[key].parent = None
        else:
            for item in self[key]:
                item.parent = None
        super().__delitem__(key)

    def insert(self, index: int, value):
        value.parent = self
        super().insert(index, value)

    def extend(self, values: Iterable):
        for v in values:
            v.parent = self
        super().extend(values)

    def is_group(self) -> bool:
        return True

    def traverse(self) -> Generator[Node, None, None]:
        "Recursively traverse all nodes and leaves of the Group tree."
        yield self
        for child in self:
            yield from child.traverse()

    def get_nested_index(self, indices: Tuple[int, ...]) -> Node:
        item: Group = self
        for idx in indices:
            item = item[idx]
        return item

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        lines = []
        lines.append(self.name)

        for n, child in enumerate(self):
            child_tree = child._render()
            lines.append('  +--' + child_tree.pop(0))
            spacer = '   ' if n == len(self) - 1 else '  |'
            lines.extend([spacer + l for l in child_tree])

        return lines


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtNodeTreeModel(QAbstractItemModel):
    def __init__(self, root: Group, parent: QWidget = None):
        super().__init__(parent)
        self.root_item = root

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

    def insertRows(self, pos: int, count: int, parent: QModelIndex) -> bool:
        parentItem = self.getItem(parent)
        if pos < 0 or pos > len(parentItem):
            return False

        self.beginInsertRows(parent, pos, pos + count - 1)
        for i in range(count):
            item = Leaf()
            parentItem.insert(pos, item)
        self.endInsertRows()

        return True

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

    def removeRows(self, pos: int, count: int, parent: QModelIndex):
        print("RemoveRows")
        parentItem = self.getItem(parent)
        if pos < 0 or (pos + count) > len(parentItem):
            return False

        self.beginRemoveRows(parent, pos, pos + count - 1)
        for i in range(count):
            parentItem.pop(pos)
        self.endRemoveRows()

        return True

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
            text.append(str(id(item)))
        mimedata.setData(self.mimeTypes()[0], pickle.dumps(data))
        mimedata.setText(" ".join(text))
        return mimedata

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

        dest_parent_item = self.getItem(parent)
        dest_tup = dest_parent_item.index_from_root()
        if destRow == -1:
            destRow = len(dest_parent_item)

        dragged_indices = pickle.loads(data.data(default_format))
        if len(dragged_indices) <= 1:
            # simpler task
            for *p, srcRow in dragged_indices:
                src_parent = self.get_nested_index(p)
                self.moveRows(src_parent, srcRow, 1, parent, destRow)
            return False

        # more complicated when moving multiple objects.
        # don't assume selection adjacency ... so move one at a time
        # need to update indices as we pop, so we keep track of the indices
        # we have previously popped
        popped: DefaultDict[Tuple[int, ...], List[int]] = defaultdict(list)
        # we iterate indices from the end first, so pop() always works
        for i, (*p, srcRow) in enumerate(
            sorted(dragged_indices, reverse=True)
        ):
            src_tup = tuple(p)  # index of parent relative to root
            src_parent = self.get_nested_index(src_tup)

            # we need to decrement the srcRow by 1 for each time we have
            # previously pulled items out from in front of the srcRow
            sdec = sum(map(lambda x: x <= srcRow, popped.get(src_tup, [])))
            # if item is being moved within the same parent,
            # we need to increase the srcRow by 1 for each time we have
            # previously inserted items in front of the srcRow
            if src_tup == dest_tup:
                sdec -= (destRow <= srcRow) * i
            # we need to decrement the destRow by 1 for each time we have
            # previously pulled items out from in front of the destRow
            ddec = sum(map(lambda x: x <= destRow, popped.get(dest_tup, [])))

            self.moveRows(src_parent, srcRow - sdec, 1, parent, destRow - ddec)
            popped[src_tup].append(srcRow)

        # If we return true, removeRows is called!?
        return False

    def get_nested_index(self, indices: Tuple[int, ...]) -> QModelIndex:
        parentIndex = QModelIndex()
        for idx in indices:
            parentIndex = model.index(idx, 0, parentIndex)
        return parentIndex


class QtNodeTree(QTreeView):
    def __init__(self, root, parent: QWidget = None):
        super().__init__(parent)
        self.setModel(QtNodeTreeModel(root, self))
        self.setHeaderHidden(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.setStyleSheet(
            "QTreeView::item {" "padding: 15px;" "color: red;" "}"
        )


if __name__ == '__main__':
    from napari import gui_qt

    with gui_qt():
        tip = Leaf(name='tip')
        lg2 = Group(name="g2", children=[Leaf(name='2')])
        lg1 = Group(
            name="g1", children=[lg2, Leaf(name='3'), tip, Leaf(name='1')]
        )
        root = Group(
            name="root",
            children=[
                lg1,
                Leaf(name='4'),
                Leaf(name='5'),
                Leaf(name='6'),
                Leaf(name='7'),
                Leaf(name='8'),
                Leaf(name='9'),
            ],
        )
        tree = QtNodeTree(root)
        model = tree.model()
        root.events.connect(lambda e: model.layoutChanged.emit())
        tree.show()
