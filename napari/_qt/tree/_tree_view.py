from qtpy.QtWidgets import (
    QAbstractItemView,
    QTreeView,
    QWidget,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from qtpy.QtCore import QItemSelection, QItemSelectionModel, QModelIndex
from qtpy.QtGui import QPainter

from ...utils.tree.group import Group, NestedIndex
from ...layers.layergroup import LayerGroup
from ._tree_model import QtNodeTreeModel, QtLayerTreeModel


class QtNodeTreeView(QTreeView):
    def __init__(self, root: Group = None, parent: QWidget = None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setStyleSheet(r"QTreeView::item {padding: 4px;}")
        if root is not None:
            self.setRoot(root)

    def setRoot(self, root: Group):
        self.setModel(QtNodeTreeModel(root, self))


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

    def setRoot(self, root):
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

    def _select(self, nested_index: NestedIndex, selected=True):
        idx = self.model().nestedIndex(nested_index)
        s = getattr(QItemSelectionModel, 'Select' if selected else 'Deselect')
        self.selectionModel().select(idx, s)
