from qtpy.QtWidgets import QTreeView
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt
from napari.layers import LayerGroup, Layer
import weakref


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtLayerTreeModel(QAbstractItemModel):
    def __init__(self, layergroup: LayerGroup = None, parent=None):
        super().__init__(parent)
        self._root = layergroup or LayerGroup()
        self._root.events.changed.connect(self._refresh)

    def rowCount(self, index: QModelIndex) -> int:
        if index.isValid():
            return len(self.itemAt(index))
        return len(self._root)

    def columnCount(self, QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = Qt.DisplayRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            # TODO: not supposed to use internal pointer for data
            return str(self.itemAt(index).name)
        return None

    def _refresh(self, e=None):
        print()
        super().beginInsertRows(QModelIndex(), 0, 0)
        super().endInsertRows()

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        return super().flags(index)

    def index(
        self, row: int, col: int, parent: QModelIndex = None
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

        parent: Layer = self.itemAt(index).parent

        if not parent or parent == self._root:
            return QModelIndex()

        return super().createIndex(parent.row, 0, weakref.ref(parent))

    def insertRow(self, row: int, parent: QModelIndex = None) -> bool:
        parent_item: LayerGroup = self.itemAt(parent)
        if not parent_item:
            return False

        super().beginInsertRows(parent, row, row)
        # parent_item.insert(row)
        super().endInsertRows()
        return True

    def itemAt(self, index: QModelIndex) -> Layer:
        if index and index.isValid():
            item = index.internalPointer()()
            if item is not None:
                return item
        return self._root


class QtLayerTree(QTreeView):
    def __init__(self, layergroup, parent=None):
        super().__init__(parent)
        self.layergroup = layergroup
        self.setModel(QtLayerTreeModel(layergroup, self))
