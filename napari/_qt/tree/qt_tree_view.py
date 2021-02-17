from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QTreeView

from .qt_tree_model import QtNodeTreeModel

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ...utils.tree.group import Group


class QtNodeTreeView(QTreeView):
    def __init__(self, root: Group = None, parent: QWidget = None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragDropMode(QTreeView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QTreeView.ExtendedSelection)
        if root is not None:
            self.setRoot(root)

    def setRoot(self, root: Group):
        self.setModel(QtNodeTreeModel(root, self))
