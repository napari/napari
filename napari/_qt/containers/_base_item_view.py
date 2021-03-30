from __future__ import annotations

from itertools import chain, repeat
from typing import TYPE_CHECKING, Generic, TypeVar

from qtpy.QtCore import QItemSelection, QModelIndex, Qt
from qtpy.QtWidgets import QAbstractItemView

from ._factory import create_model

ItemType = TypeVar("ItemType")

if TYPE_CHECKING:
    from qtpy.QtGui import QKeyEvent

    from ...utils.events import Event


class _BaseEventedItemView(QAbstractItemView, Generic[ItemType]):
    def setRoot(self, root):
        self._root = root
        self.setModel(create_model(root, self))

        # connect selection events
        root.selection.events.changed.connect(self._on_py_selection_change)
        root.selection.events._current.connect(self._on_py_current_change)
        self._sync_selection_models()

    def _on_py_current_change(self, event: Event):
        """The python model current item has changed.  Update the Qt view."""
        sm = self.selectionModel()
        if not event.value:
            sm.clearCurrentIndex()
        else:
            idx = self.model().findIndex(event.value)
            sm.setCurrentIndex(idx, sm.Current)

    def _on_py_selection_change(self, event: Event):
        """The python model selection has changed.  Update the Qt view."""
        sm = self.selectionModel()
        for is_selected, idx in chain(
            zip(repeat(sm.Select), event.added),
            zip(repeat(sm.Deselect), event.removed),
        ):
            model_idx = self.model().findIndex(idx)
            if model_idx.isValid():
                sm.select(model_idx, is_selected)

    def _sync_selection_models(self):
        """Clear and re-sync the Qt selection view from the python selection."""
        sel_model = self.selectionModel()
        selection = QItemSelection()
        for i in self._root.selection:
            idx = self.model().findIndex(i)
            selection.select(idx, idx)
        sel_model.select(selection, sel_model.ClearAndSelect)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        """delete items with delete key."""
        if e.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            for i in self.selectionModel().selectedIndexes():
                self._root.remove(i.internalPointer())
        return super().keyPressEvent(e)

    def currentChanged(self, current: QModelIndex, previous: QModelIndex):
        """The Qt current item has changed. Update the python model."""
        self._root.selection._current = current.internalPointer()
        return super().currentChanged(current, previous)

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        """The Qt Selection has changed. Update the python model."""
        s = self._root.selection
        s.difference_update(i.internalPointer() for i in deselected.indexes())
        s.update(i.internalPointer() for i in selected.indexes())
        return super().selectionChanged(selected, deselected)
