from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QTreeView

from ._base_item_view import _BaseItemView
from .qt_tree_model import QtNodeTreeModel

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ...utils.tree import Group, Node


class QtNodeTreeView(QTreeView, _BaseItemView):
    model_class = QtNodeTreeModel
    _root: Group[Node]

    def __init__(self, root: Group[Node], parent: QWidget = None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragDropMode(QTreeView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QTreeView.ExtendedSelection)
        self.setRoot(root)

    def setRoot(self, root: Group[Node]):
        self._root = root
        model = self.model_class(root, self)
        self.setModel(model)

        # connect model events
        model.rowsRemoved.connect(self._redecorate_root)
        model.rowsInserted.connect(self._redecorate_root)
        self._redecorate_root()

        # connect selection events
        root.selection.events.changed.connect(self._on_py_selection_change)
        root.selection.events.current.connect(self._on_py_current_change)
        self._sync_selection_models()

    def model(self) -> QtNodeTreeModel[Node]:
        return super().model()

    def _redecorate_root(self, parent=None, *_):
        """Add a branch/arrow column only if there are Groups in the root.

        This makes the tree fall back to looking like a simple list if there
        are no groups in the root level.
        """
        if not parent or not parent.isValid():
            self.setRootIsDecorated(self.model().hasGroups())
