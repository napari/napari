import pickle
import weakref
from typing import List, Union

from napari.layers import Layer, LayerGroup
from qtpy.QtCore import (
    QAbstractItemModel,
    QMimeData,
    QModelIndex,
    Qt,
    QItemSelection,
)
from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget


class Placeholder:
    def __init__(self, parent=None):
        self.parent = parent
        self.name = 'placeholder'


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtLayerTreeModel(QAbstractItemModel):
    ID_ROLE = 250

    def __init__(self, layergroup: LayerGroup = None, parent: QWidget = None):
        super().__init__(parent)
        self._root = layergroup if layergroup is not None else LayerGroup()
        self._root.events.added.connect(self._on_added)  # type: ignore
        self._root.events.removed.connect(self._on_removed)  # type: ignore

    def _on_added(self, event):
        """Notify view when data is added to the model."""
        if not event.sources:
            return

        source_index = self.indexFromItem(event.sources[0])
        for idx, item in event.value:
            super().beginInsertRows(source_index, idx, idx)
            super().endInsertRows()
            self.dataChanged.emit(source_index, source_index)

    def _on_removed(self, event):
        """Notify view when data is removed from the model."""
        if not event.sources:
            return

        source_index = self.indexFromItem(event.sources[0])
        idx, item = event.value
        super().beginRemoveRows(source_index, idx, idx)
        super().endRemoveRows()
        self.dataChanged.emit(source_index, source_index)

    def rowCount(self, index: QModelIndex) -> int:
        if index.isValid():
            return len(self.itemFromIndex(index))
        return len(self._root)

    def columnCount(self, QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = Qt.DisplayRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        if role == self.ID_ROLE:
            print("ID ROLE!")
            return id(self.itemFromIndex(index))
        if role == Qt.DisplayRole:
            return str(self.itemFromIndex(index).name)
        if role == Qt.UserRole:
            return 1
        return None

    def flags(self, index: QModelIndex) -> Union[Qt.ItemFlag, Qt.ItemFlags]:
        """Returns the item flags for the given index.

        The base class implementation returns a combination of flags that
        enables the item (ItemIsEnabled) and allows it to be selected
        (ItemIsSelectable).
        """
        if not index.isValid():
            return Qt.ItemIsDropEnabled
        # https://doc.qt.io/qt-5/qt.html#ItemFlag-enum
        item = self.itemFromIndex(index)

        base_flags = (
            Qt.ItemIsSelectable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEnabled
            | Qt.ItemIsDropEnabled
        )
        if isinstance(item, LayerGroup):
            return base_flags | Qt.ItemIsAutoTristate
        return base_flags | Qt.ItemNeverHasChildren

    def index(
        self, row: int, col: int = 0, parent: QModelIndex = None
    ) -> QModelIndex:
        """Return index of the item specified by row, column and parent index.

        Given a model index for a parent item, this function allows views and
        delegates to access children of that item. If no valid child item -
        corresponding to the specified row column, and parent model index,
        can be found, the function must return QModelIndex(), which is an
        invalid model index.
        """
        parent_item = self.itemFromIndex(parent)

        if not super().hasIndex(row, col, parent or QModelIndex()):
            return QModelIndex()

        # When reimplementing this function in a subclass, call createIndex()
        # to generate model indexes that other components can use to refer to
        # items in your model.
        try:
            child = parent_item[row]  # type: ignore
            return super().createIndex(row, col, weakref.ref(child))
        except (TypeError, IndexError):  # regular layer not subscriptable
            return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Returns the parent of the model item with the given index.

        If the item has no parent, an invalid QModelIndex is returned.
        """
        if not index.isValid():
            return QModelIndex()

        parent = self.itemFromIndex(index).parent

        if not parent or parent == self._root:
            return QModelIndex()

        return super().createIndex(parent.row, 0, weakref.ref(parent))

    def itemFromIndex(self, index: QModelIndex) -> Layer:
        """Return Layer (or LayerGroup) associated with the given index."""
        if index and index.isValid():
            item = index.internalPointer()()
            if item is not None:
                return item
        return self._root

    def _itemFromID(self, id: int) -> Layer:
        return self.itemFromIndex(self._indexFromID(id))

    def _indexFromID(self, id: int) -> QModelIndex:
        hits = self.match(
            self.index(0),
            self.ID_ROLE,
            id,
            1,
            Qt.MatchExactly | Qt.MatchRecursive,
        )
        if hits:
            return hits[0]
        raise IndexError(f"ID {id} not found in model")

    def indexFromItem(self, item: Layer) -> QModelIndex:
        """Return QModelIndex for an item in the model (recursive).

        Raises
        ------
        IndexError
            If the item is not in the model
        """
        if item == self._root:
            return QModelIndex()
        try:
            return self._indexFromID(id(item))
        except IndexError:
            raise IndexError(f"item {item} not found in model")

    def canDropMimeData(self, *args):
        return isinstance(self.itemFromIndex(args[-1]), LayerGroup)

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
        data = [
            (i.row(), i.column(), self.data(i, self.ID_ROLE)) for i in indices
        ]
        mimedata.setData('application/x-layertree', pickle.dumps(data))
        mimedata.setText(" ".join(str(x[2]) for x in data))
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

        When row and col are -1 it means that the dropped data should be
        considered as dropped directly on parent. Usually this will mean
        appending the data as child items of parent. If row and col are
        greater than or equal zero, it means that the drop occurred just before
        the specified row and col in the specified parent.

        https://doc-snapshots.qt.io/qt5-5.12/model-view-programming.html#drag-and-drop-support-and-mime-type-handling

        """
        # ids = [self._indexFromID(i) for i in data.text().split()]
        # return super().dropMimeData(data, action, row, col, parent)
        if not data or action != Qt.MoveAction:
            return False
        default_format = self.mimeTypes()[0]
        if not data.hasFormat(default_format):
            return False

        dragged_items = pickle.loads(data.data(default_format))
        new_parent = self.itemFromIndex(parent)
        if row == col == -1:
            # appending to new parent
            for cur_row, cur_col, item_id in reversed(dragged_items):
                item = self._itemFromID(item_id)
                item.parent.remove(item)
                new_parent.append(item)  # type: ignore
        else:
            for cur_row, cur_col, item_id in reversed(dragged_items):
                item = self._itemFromID(item_id)
                if new_parent == item.parent:
                    # internal move
                    item.parent.move(cur_row, row)
                else:
                    # moving to precise position in new parent
                    item.parent.remove(item)
                    new_parent.insert(row, item)  # type: ignore
        return True

    # def removeRows(self, row: int, count: int, parent: QModelIndex) -> bool:
    #     print("remove", row, count, self.itemFromIndex(parent).name)
    #     super().beginRemoveRows(parent, row, row + count - 1)
    #     item = self.itemFromIndex(parent)
    #     del item[row]
    #     super().endRemoveRows()
    #     return True

    # def setItemData(self, index: QModelIndex, roles: dict) -> bool:
    #     _id = roles.get(self.ID_ROLE)
    #     item = self._root.find_id(_id)
    #     if item is None:
    #         raise IndexError("No ITEM!")

    #     parent = self.itemFromIndex(self.parent(index))
    #     parent._list[index.row()] = item
    #     print(f"set it to {item.name}")
    #     self.dataChanged.emit(index, index)
    #     return True

    # def insertRows(self, row: int, count: int, parent: QModelIndex) -> bool:
    #     print("insert")
    #     item = self.itemFromIndex(parent)
    #     if row < 0 or row > len(item):
    #         return False

    #     self.beginInsertRows(parent, row, row + count - 1)
    #     for i in range(count):
    #         p = Points(name='d')
    #         with item.events.blocker():
    #             item.insert(row, p)
    #         row += 1
    #     self.endInsertRows()
    #     return True

    def setSelection(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        for idx in selected.indexes():
            self.itemFromIndex(idx).selected = True
        for idx in deselected.indexes():
            self.itemFromIndex(idx).selected = False


class QtLayerTree(QTreeView):
    def __init__(self, layergroup: LayerGroup = None, parent: QWidget = None):
        super().__init__(parent)
        _model = QtLayerTreeModel(layergroup, self)
        self.setModel(_model)
        self.setHeaderHidden(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selectionModel().selectionChanged.connect(_model.setSelection)


if __name__ == '__main__':
    from napari.layers import Points, Shapes, LayerGroup
    from napari import gui_qt

    with gui_qt():
        pts = Points()
        lg2 = LayerGroup([Shapes(name='s1')], name="lg2")
        lg1 = LayerGroup([lg2, Points(name='p1'), pts], name="lg1")
        root = LayerGroup(
            [lg1, Points(name='p2'), Shapes(name='s2'), Points(name='p3')],
            name="root",
        )
        tree = QtLayerTree(root)
        model = tree.model()
        tree.show()
