from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget
from qtpy.QtCore import QItemSelection, QItemSelectionModel

from ...utils.tree.group import Group, NestedIndex
from ...layers.layergroup import LayerGroup
from ._tree_model import QtNodeTreeModel


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


class QtLayerTreeView(QtNodeTreeView):
    def setRoot(self, root: LayerGroup):
        super().setRoot(root)
        root.events.selected.connect(lambda e: self._select(e.index, e.value))
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
