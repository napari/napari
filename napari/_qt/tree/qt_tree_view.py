from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget

from ...utils.tree.group import Group
from .qt_tree_model import QtNodeTreeModel


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
