from qtpy.QtWidgets import QTreeView
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt
from napari.layers import LayerGroup, Layer
import weakref


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtLayerTreeModel(QAbstractItemModel):
    def __init__(self, layergroup: LayerGroup = None, parent=None):
        super().__init__(parent)
        self._root = layergroup or LayerGroup()
        self._root._children.events.added.connect(self._add)

    def _add(self, event):
        print(vars(event))

    def rowCount(self, index: QModelIndex) -> int:
        if index.isValid():
            return len(self.getItem(index))
        return len(self._root)

    def columnCount(self, QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = Qt.DisplayRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            # TODO: not supposed to use internal pointer for data
            return str(self.getItem(index).name)
        return None

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

        parent_item = self.getItem(index).parent

        if not parent_item or parent_item == self._root:
            return QModelIndex()

        return super().createIndex(
            parent_item.row, 0, weakref.ref(parent_item)
        )

    def insertRow(self, row: int, parent: QModelIndex = None) -> bool:
        parent_item: LayerGroup = self.getItem(parent)
        if not parent_item:
            return False

        super().beginInsertRows(parent, row, row)
        parent_item.insert(row)
        super().endInsertRows()
        return True

    def getItem(self, index: QModelIndex) -> Layer:
        if index and index.isValid():
            item = index.internalPointer()()
            if item:
                return item
        return self._root


class QtLayerTree(QTreeView):
    def __init__(self, layergroup, parent=None):
        super().__init__(parent)
        self.layergroup = layergroup
        self.setModel(QtLayerTreeModel(layergroup, self))
