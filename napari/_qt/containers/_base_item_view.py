from __future__ import annotations

from itertools import chain, repeat
from typing import TYPE_CHECKING, Generic, TypeVar

from qtpy.QtCore import QItemSelection, QModelIndex, Qt
from qtpy.QtWidgets import QAbstractItemView

from ._factory import create_model

ItemType = TypeVar("ItemType")

if TYPE_CHECKING:
    from qtpy.QtCore import QAbstractItemModel
    from qtpy.QtGui import QKeyEvent

    from ...utils.events import Event
    from ...utils.events.containers import SelectableEventedList
    from ._base_item_model import _BaseEventedItemModel


class _BaseEventedItemView(Generic[ItemType]):
    """A QAbstractItemView mixin desigend to work with `SelectableEventedList`.

    :class:`~napari.utils.events.SelectableEventedList` is our pure python
    model of a mutable sequence that supports the concept of "currently
    selected/active items".  It emits events when the list is altered (e.g.,
    by appending, inserting, removing items), or when the selection model is
    altered.

    This class is an adapter between that interface and Qt's
    `QAbstractItemView` interface (see `Qt Model/View Programming
    <https://doc.qt.io/qt-5/model-view-programming.html>`_). It allows python
    users to interact with the list in the "usual" python ways, while updating
    any Qt Views that may be connected, and also updates the python list object
    if any GUI events occur in the view.

    For a "plain" (flat) list, use the
    :class:`napari._qt.containers.QtListView` subclass.
    For a nested list-of-lists using the Group/Node classes, use the
    :class:`napari._qt.containers.QtNodeTreeView` subclass.

    For convenience, the :func:`napari._qt.containers.create_view` factory
    function will return the appropriate `_BaseEventedItemView` instance given
    a python `EventedList` object.
    """

    # ########## Reimplemented Public Qt Functions ##################

    def model(self) -> _BaseEventedItemModel[ItemType]:  # for type hints
        return super().model()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        """Delete items with delete key."""
        if e.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            self._root.remove_selected()
            return
        return super().keyPressEvent(e)

    def currentChanged(
        self: QAbstractItemView, current: QModelIndex, previous: QModelIndex
    ):
        """The Qt current item has changed. Update the python model."""
        self._root.selection._current = current.data(Qt.UserRole)
        return super().currentChanged(current, previous)

    def selectionChanged(
        self: QAbstractItemView,
        selected: QItemSelection,
        deselected: QItemSelection,
    ):
        """The Qt Selection has changed. Update the python model."""
        s = self._root.selection
        s.difference_update(i.data(Qt.UserRole) for i in deselected.indexes())
        s.update(i.data(Qt.UserRole) for i in selected.indexes())
        return super().selectionChanged(selected, deselected)

    # ###### Non-Qt methods added for SelectableEventedList Model ############

    def setRoot(self, root: SelectableEventedList[ItemType]):
        """Call during __init__, to set the python model."""
        self._root = root
        self.setModel(create_model(root, self))

        # connect selection events
        root.selection.events.changed.connect(self._on_py_selection_change)
        root.selection.events._current.connect(self._on_py_current_change)
        self._sync_selection_models()

    def _on_py_current_change(self, event: Event):
        """The python model current item has changed. Update the Qt view."""
        sm = self.selectionModel()
        if not event.value:
            sm.clearCurrentIndex()
        else:
            idx = index_of(self.model(), event.value)
            sm.setCurrentIndex(idx, sm.Current)

    def _on_py_selection_change(self, event: Event):
        """The python model selection has changed. Update the Qt view."""
        sm = self.selectionModel()
        for is_selected, idx in chain(
            zip(repeat(sm.Select), event.added),
            zip(repeat(sm.Deselect), event.removed),
        ):
            model_idx = index_of(self.model(), idx)
            if model_idx.isValid():
                sm.select(model_idx, is_selected)

    def _sync_selection_models(self):
        """Clear and re-sync the Qt selection view from the python selection."""
        sel_model = self.selectionModel()
        selection = QItemSelection()
        for i in self._root.selection:
            idx = index_of(self.model(), i)
            selection.select(idx, idx)
        sel_model.select(selection, sel_model.ClearAndSelect)


def index_of(model: QAbstractItemModel, obj: ItemType) -> QModelIndex:
    """Find the `QModelIndex` for a given object in the model."""
    fl = Qt.MatchExactly | Qt.MatchRecursive
    hits = model.match(
        model.index(0, 0, QModelIndex()), Qt.UserRole, obj, hits=1, flags=fl
    )
    return hits[0] if hits else QModelIndex()
