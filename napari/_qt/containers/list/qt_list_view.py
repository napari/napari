from __future__ import annotations

from itertools import chain, repeat
from typing import TYPE_CHECKING, Generic, TypeVar

from qtpy.QtCore import QItemSelection, QModelIndex, Qt
from qtpy.QtWidgets import QListView

from .qt_list_model import QtListModel

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ....utils.events import Event
    from ....utils.events.containers import SelectableEventedList

ItemType = TypeVar("ItemType")


class QtListView(QListView, Generic[ItemType]):
    model_class = QtListModel
    _list: SelectableEventedList[ItemType]

    def __init__(
        self, root: SelectableEventedList[ItemType], parent: QWidget = None
    ):
        super().__init__(parent)
        self.setDragDropMode(QListView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setRoot(root)

    def setRoot(self, root: SelectableEventedList[ItemType]):
        self._list = root
        model = self.model_class(root, self)
        self.setModel(model)
        # connect selection events
        root.selection.events.changed.connect(self._on_py_selection_change)
        root.selection.events.current.connect(self._on_py_current_change)
        self._sync_selection_models()

    def model(self) -> QtListModel[ItemType]:
        return super().model()

    def _sync_selection_models(self):
        """Clear and re-sync the Qt selection view from the python selection."""
        sel_model = self.selectionModel()
        selection = QItemSelection()
        for i in self._list.selection:
            idx = self.model().findIndex(i)
            selection.select(idx, idx)
        sel_model.select(selection, sel_model.ClearAndSelect)

    def keyPressEvent(self, event) -> None:
        """delete items with delete key."""
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            for i in self.selectionModel().selectedIndexes():
                self._list.remove(self.model().getItem(i))
        return super().keyPressEvent(event)

    def currentChanged(self, current: QModelIndex, previous: QModelIndex):
        """The Qt current item has changed. Update the python model."""
        self._list.selection.current = self.model().getItem(current)
        return super().currentChanged(current, previous)

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        """The Qt Selection has changed. Update the python model."""
        s = self._list.selection
        s.difference_update(
            self.model().getItem(i) for i in deselected.indexes()
        )
        s.update(self.model().getItem(i) for i in selected.indexes())
        return super().selectionChanged(selected, deselected)

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
