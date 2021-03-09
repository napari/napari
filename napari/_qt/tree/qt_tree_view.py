from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QItemSelection, QModelIndex, Qt
from qtpy.QtWidgets import QTreeView

from .qt_tree_model import QtNodeTreeModel

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ...utils.events import Event
    from ...utils.tree import Group, Node


class QtNodeTreeView(QTreeView):
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
        root.selection.events.connect(self._on_py_selection_model_event)
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

    def _sync_selection_models(self):
        """Clear and re-sync the Qt selection view from the python selection."""
        sel_model = self.selectionModel()
        selection = QItemSelection()
        for i in self._root.selection:
            idx = self.model().findIndex(i)
            selection.select(idx, idx)
        sel_model.select(selection, sel_model.ClearAndSelect)

    def keyPressEvent(self, event) -> None:
        """delete items with delete key."""
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            for i in self.selectionModel().selectedIndexes():
                self._root.remove(i.internalPointer())
        return super().keyPressEvent(event)

    def currentChanged(self, current: QModelIndex, previous: QModelIndex):
        """The Qt current item has changed. Update the python model."""
        self._root.selection.current = current.internalPointer()
        return super().currentChanged(current, previous)

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        """The Qt Selection has changed. Update the python model."""
        s = self._root.selection
        s.difference_update(i.internalPointer() for i in deselected.indexes())
        s.update(i.internalPointer() for i in selected.indexes())
        return super().selectionChanged(selected, deselected)

    def _on_py_selection_model_event(self, event: Event):
        """The python model selection has changed.  Update the Qt view."""
        sel_model = self.selectionModel()
        if event.type == 'current':
            if not event.value:
                sel_model.clearCurrentIndex()
            else:
                idx = self.model().findIndex(event.value)
                sel_model.setCurrentIndex(idx, sel_model.Current)
            return
        t = sel_model.Select if event.type == 'added' else sel_model.Deselect
        for idx in event.value:
            model_idx = self.model().findIndex(idx)
            if not model_idx.isValid():
                continue
            sel_model.select(model_idx, t)
