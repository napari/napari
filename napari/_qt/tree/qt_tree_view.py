from __future__ import annotations

from itertools import chain, repeat
from typing import TYPE_CHECKING

from qtpy.QtCore import QItemSelection, QItemSelectionModel, QModelIndex
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
        self._root: Group[Node] = root
        self.setModel(self.model_class(root, self))

        # connect model events
        self.model().rowsRemoved.connect(self._redecorate_root)
        self.model().rowsInserted.connect(self._redecorate_root)
        self._redecorate_root()

        # connect selection events
        root.selection.events.changed.connect(self._on_py_selection_change)
        root.selection.events._current.connect(self._on_py_current_change)

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
        sel_model: QItemSelectionModel = self.selectionModel()
        selection = QItemSelection()
        for i in self._root.selection:
            idx = self.model().findIndex(i)
            selection.select(idx, idx)
        sel_model.select(selection, sel_model.ClearAndSelect)

    def currentChanged(self, current: QModelIndex, previous: QModelIndex):
        """The Qt current item has changed. Update the python model."""
        item = current.internalPointer()
        self._root.selection._current = item or None
        return super().currentChanged(current, previous)

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        """The Qt Selection has changed. Update the python model."""
        s = self._root.selection
        s.difference_update(i.internalPointer() for i in deselected.indexes())
        s.update(i.internalPointer() for i in selected.indexes())
        return super().selectionChanged(selected, deselected)

    def _on_py_current_change(self, event: Event):
        """The python model current item has changed. Update the Qt view."""
        sm = self.selectionModel()
        if not event.value:
            sm.clearCurrentIndex()
        else:
            idx = self.model().findIndex(event.value)
            sm.setCurrentIndex(idx, sm.Current)

    def _on_py_selection_change(self, event: Event):
        """The python model selection has changed. Update the Qt view."""
        sm = self.selectionModel()
        for is_selected, obj in chain(
            zip(repeat(sm.Select), event.added),
            zip(repeat(sm.Deselect), event.removed),
        ):
            model_idx = self.model().findIndex(obj)
            if model_idx.isValid():
                sm.select(model_idx, is_selected)
