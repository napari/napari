from typing import TYPE_CHECKING, cast
from qtpy.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt
from .qt_tree_model import QtNodeTreeModel

if TYPE_CHECKING:
    from napari.layers import Layer, LayerGroup
    from qtpy.QtWidgets import QWidget


class QtLayerTreeModel(QtNodeTreeModel):
    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.setRoot(root)

    def getItem(self, index: QModelIndex) -> Layer:
        # TODO: this cast should be fixed by making QtNodeTreeModel Generic.
        return cast(Layer, super().getItem(index))

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        item = self.getItem(index)
        if role == Qt.DisplayRole:
            return item.name
        if role == Qt.CheckStateRole:
            return item.visible
        if role == Qt.UserRole:
            return self.getItem(index)
        return None
