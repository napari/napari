from __future__ import annotations
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, Tuple, Generator

from qtpy.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    QMimeData,
)
from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget
from napari.utils.list._evented_list import EventedList
from typing import MutableSequence


class Component(ABC):
    def __init__(self, name='Component', children=None) -> None:
        self._children: MutableSequence[Component] = EventedList()
        self._parent: Optional[Component] = None
        self.name = name
        for child in children or []:
            self.append(child)

    def __len__(self) -> int:
        return len(self._children)

    @property
    def parent(self) -> Optional[Component]:
        return self._parent

    @parent.setter
    def parent(self, parent: Component):
        self._parent = parent

    def clear(self) -> None:
        pass

    def append(self, component: Component):
        pass

    def insert(self, index: int, component: Component):
        pass

    def remove(self, component: Component):
        pass

    def pop(self, index: int = -1) -> Component:
        pass

    def is_composite(self) -> bool:
        return False

    @abstractmethod
    def operation(self) -> str:
        pass

    def __iter__(self):
        yield self

    def index_in_parent(self) -> int:
        if self.parent is not None:
            return self.parent._children.index(self)
        return 0

    def index_from_root(self) -> Tuple[int, ...]:
        indices: List[int] = []
        item = self
        while item.parent is not None:
            indices.insert(0, item.index_in_parent())
            item = item.parent
        return tuple(indices)

    def get_nested_index(self, indices: Tuple[int, ...]) -> Component:
        item = self
        for idx in indices:
            item = item._children[idx]
        return item

    def traverse(self):
        yield self

    def __str__(self):
        """Render ascii tree string representation of this component"""
        return "\n".join(self._render())

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        lines = []
        lines.append(self.name)

        for n, child in enumerate(self._children):
            child_tree = child._render()
            lines.append('  +--' + child_tree.pop(0))
            spacer = '   ' if n == len(self) - 1 else '  |'
            lines.extend([spacer + l for l in child_tree])

        return lines


class Leaf(Component):
    def operation(self) -> str:
        return "Leaf"


class Composite(Component):
    def __iter__(self):
        yield from iter(self._children)

    def clear(self) -> None:
        while self._children:
            self.pop()

    def append(self, component: Component):
        self._children.append(component)
        component.parent = self

    def insert(self, index: int, component: Component):
        self._children.insert(index, component)
        component.parent = self

    def remove(self, component: Component):
        self._children.remove(component)
        component.parent = None

    def pop(self, index: int = -1) -> Component:
        component = self._children.pop(index)
        component.parent = None
        return component

    def is_composite(self) -> bool:
        return True

    def operation(self) -> str:
        results = [child.operation() for child in self._children]
        return f"Branch({'+'.join(results)})"

    def traverse(self) -> Generator[Component, None, None]:
        "Recursively traverse all nodes and leaves of the Composite tree."
        yield self
        for child in self:
            yield from child.traverse()

    def __getitem__(self, key) -> Component:
        return self._children[key]


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtLayerTreeModel(QAbstractItemModel):
    def __init__(self, root: Composite, parent: QWidget = None):
        super().__init__(parent)
        self.root_item = root

    def canDropMimeData(self, *args):
        return self.getItem(args[-1]).is_composite()

    def columnCount(self, parent: QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        """Return data stored under ``role`` for the item at ``index``."""
        item = self.getItem(index)
        if role == Qt.DisplayRole:
            return str(item.name)
        return None

    def getItem(self, index: QModelIndex) -> Component:
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
        if item.is_composite():
            return base_flags
        return base_flags | Qt.ItemNeverHasChildren

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = None
    ) -> QModelIndex:
        """Return index of the item specified by row, column and parent index.
        """
        if not self.hasIndex(row, column, parent or QModelIndex()):
            return QModelIndex()

        try:
            childItem = self.getItem(parent)._children[row]
            return self.createIndex(row, column, childItem)
        except (AttributeError, IndexError):
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
            item = srcParentItem.pop(sourceRow)
            destParentItem.insert(destinationChild, item)
            print(
                f"move {item.name} from {srcParentItem.name}:{sourceRow} to {destParentItem.name}:{destinationChild}"
            )
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

    def setData(
        self, index: QModelIndex, value: Any, role: Qt.ItemDataRole
    ) -> bool:
        print("setData", index, value, role)
        return super().setData(index, value, role)

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
        row: int,
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
        for nested_idx in dragged_indices:
            *p, sourceRow = nested_idx
            sourceParent = self.get_nested_inded(p)
            self.moveRows(sourceParent, sourceRow, 1, parent, row)

        # If we return true, removeRows is called!?
        return False

    def get_nested_inded(self, indices: Tuple[int, ...]) -> QModelIndex:
        parentIndex = model.index(0)
        for idx in indices:
            parentIndex = model.index(idx, 0, parentIndex)
        return parentIndex


class QtLayerTree(QTreeView):
    def __init__(self, root, parent: QWidget = None):
        super().__init__(parent)
        self.setModel(QtLayerTreeModel(root, self))
        self.setHeaderHidden(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)


if __name__ == '__main__':
    from napari import gui_qt

    with gui_qt():
        tip = Leaf(name='tip')
        lg2 = Composite(name="lg2", children=[Leaf(name='s1')])
        lg1 = Composite(name="lg1", children=[lg2, Leaf(name='p1'), tip])
        root = Composite(
            name="root",
            children=[lg1, Leaf(name='p2'), Leaf(name='s2'), Leaf(name='p3')],
        )
        tree = QtLayerTree(root)
        model = tree.model()
        model.rowsMoved.connect(lambda x: print(root))
        tree.show()
