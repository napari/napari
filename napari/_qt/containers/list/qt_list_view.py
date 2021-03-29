from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from qtpy.QtWidgets import QListView

from .._base_item_view import _BaseItemView
from .qt_list_model import QtListModel

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ....utils.events.containers import SelectableEventedList

ItemType = TypeVar("ItemType")


class QtListView(QListView, _BaseItemView[ItemType]):
    model_class = QtListModel
    _root: SelectableEventedList[ItemType]

    def __init__(
        self, root: SelectableEventedList[ItemType], parent: QWidget = None
    ):
        super().__init__(parent)
        self.setDragDropMode(QListView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setRoot(root)

    def model(self) -> QtListModel[ItemType]:
        return super().model()
