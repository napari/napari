from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QItemSelection, QItemSelectionModel, QModelIndex, Qt
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QWidget

from .qt_tree_model import QtNodeTreeModel
from .qt_tree_view import QtNodeTreeView

if TYPE_CHECKING:
    from ...layers import Layer
    from ...layers.layergroup import LayerGroup
    from ...utils.events.containers._nested_list import MaybeNestedIndex


class QtLayerTreeModel(QtNodeTreeModel):
    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.setRoot(root)

    def getItem(self, index: QModelIndex) -> Layer:
        # TODO: this ignore should be fixed by making QtNodeTreeModel Generic.
        return super().getItem(index)  # type: ignore

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


class LayerDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ):
        # layer = index.internalPointer()
        super().paint(painter, option, index)


class QtLayerTreeView(QtNodeTreeView):
    def __init__(self, root: LayerGroup = None, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())

    def setRoot(self, root: LayerGroup):
        self.setModel(QtLayerTreeModel(root, self))
        root.events.selection.connect(lambda e: self._select(e.index, e.value))
        # initialize selection model
        for child in root.traverse():
            selected = getattr(child, 'selected', False)
            self._select(child.index_from_root(), selected)

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        model = self.model()
        for q_index in selected.indexes():
            model.getItem(q_index).selected = True
        for q_index in deselected.indexes():
            model.getItem(q_index).selected = False
        return super().selectionChanged(selected, deselected)

    def _select(self, nested_index: MaybeNestedIndex, selected=True):
        idx = self.model().nestedIndex(nested_index)
        s = getattr(QItemSelectionModel, 'Select' if selected else 'Deselect')
        self.selectionModel().select(idx, s)
