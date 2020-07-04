import weakref
from typing import Union

from napari.layers import Layer, LayerGroup
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt
from qtpy.QtWidgets import QTreeView


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtLayerTreeModel(QAbstractItemModel):
    ID_ROLE = 9999

    def __init__(self, layergroup: LayerGroup = None, parent=None):
        super().__init__(parent)
        self._root = layergroup or LayerGroup()
        self._root.events.added.connect(self._on_added)
        self._root.events.removed.connect(self._on_removed)

    def _on_added(self, event):
        """Notify view when data is added to the model."""
        if not event.sources:
            return

        source_index = self.indexFromItem(event.sources[0])
        for idx, item in event.value:
            super().beginInsertRows(source_index, idx, idx)
            super().endInsertRows()

    def _on_removed(self, event):
        """Notify view when data is removed from the model."""
        if not event.sources:
            return

        source_index = self.indexFromItem(event.sources[0])
        idx, item = event.value
        super().beginRemoveRows(source_index, idx, idx)
        super().endRemoveRows()

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
            return id(self.itemFromIndex(index))
        if role == Qt.DisplayRole:
            # TODO: not supposed to use internal pointer for data
            return str(self.itemFromIndex(index).name)
        return None

    def flags(self, index: QModelIndex) -> Union[Qt.ItemFlag, Qt.ItemFlags]:
        """Returns the item flags for the given index.

        The base class implementation returns a combination of flags that
        enables the item (ItemIsEnabled) and allows it to be selected
        (ItemIsSelectable).
        """
        # https://doc.qt.io/qt-5/qt.html#ItemFlag-enum
        if not index.isValid():
            return Qt.NoItemFlags
        return super().flags(index)

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
        parent_item: Layer = None
        if not parent or not parent.isValid():
            parent_item = self._root
        else:
            parent_item = parent.internalPointer()()

        if not super().hasIndex(row, col, parent or QModelIndex()):
            return QModelIndex()

        # When reimplementing this function in a subclass, call createIndex()
        # to generate model indexes that other components can use to refer to
        # items in your model.
        try:
            child = parent_item[row]
            return super().createIndex(row, col, weakref.ref(child))
        except (TypeError, IndexError):  # regular layer not subscriptable
            return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Returns the parent of the model item with the given index.

        If the item has no parent, an invalid QModelIndex is returned.
        """
        if not index.isValid():
            return QModelIndex()

        parent: Layer = self.itemFromIndex(index).parent

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

    def indexFromItem(self, item: Layer) -> QModelIndex:
        """Return QModelIndex for an item in the model (recursive).

        Raises
        ------
        IndexError
            If the item is not in the model
        """
        if item == self._root:
            return QModelIndex()
        hits = self.match(
            self.index(0),
            self.ID_ROLE,
            id(item),
            1,
            Qt.MatchExactly | Qt.MatchRecursive,
        )
        if hits:
            return hits[0]
        raise IndexError(f"item {item} not found in model")


class QtLayerTree(QTreeView):
    def __init__(self, layergroup, parent=None):
        super().__init__(parent)
        self.layergroup = layergroup
        self.setModel(QtLayerTreeModel(layergroup, self))
        self.setHeaderHidden(True)


if __name__ == '__main__':
    from napari.layers import Points, Shapes, LayerGroup
    from napari import gui_qt

    with gui_qt():
        pts = Points()
        lg2 = LayerGroup([Shapes()])
        lg1 = LayerGroup([lg2, Points(), pts])
        root = LayerGroup([lg1, Points(), Shapes()])
        tree = QtLayerTree(root)
        model = tree.model()
        tree.show()
