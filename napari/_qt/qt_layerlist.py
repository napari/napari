from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtWidgets import (
    QListWidget,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
)
from .qt_layer_widget import QtLayerWidget
from ..utils.event import Event, EmitterGroup
from .utils import drag_with_pixmap, qt_signals_blocked


class QtLayerList(QListWidget):
    """Widget storing a list of all the layers present in the current window.

    Parameters
    ----------
    layers : napari.components.LayerList
        The layer list to track and display.

    Attributes
    ----------
    layers : napari.components.LayerList
        The layer list to track and display.
    """

    def __init__(self, layers):
        super().__init__()

        self.events = EmitterGroup(
            source=self,
            auto_connect=False,
            selected_layers=Event,
            reordered=Event,
        )

        # When the EVH refactor is fully done we can do the initialization
        # and registering of the listener outside of this class and no longer
        # pass the layers object.
        self.layers = layers
        self.layers.event_handler.register_listener(self)
        self.events.connect(self.layers.event_handler.on_change)

        self.itemSelectionChanged.connect(self._selectionChanged)

        # Enable drag and drop and widget rearrangement
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)

        # Set selection mode
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.sortItems(Qt.DescendingOrder)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip('Layer list')

        # Once EVH refactor is done, this can be moved to an initialization
        # outside of this object
        self._on_selected_layers_change(self.layers.selected)

    def _selectionChanged(self):
        """Emit an event when selection changes in list widget."""
        total = self.count() - 1
        selected = [total - self.row(item) for item in self.selectedItems()]
        self.events.selected_layers(selected)

    def _on_selected_layers_change(self, selected_layers):
        """When layers selection is changed update the layers list view

        Parmeters
        ---------
        selected_layers : list
            List of selected indices.
        """
        total = self.count() - 1
        for index in range(self.count()):
            item = self.item(index)
            item.setSelected(total - index in selected_layers)

    def _on_added_change(self, value):
        """Insert widget for layer at desired location.

        Parameters
        ----------
        value : 2-tuple
            Tuple of layer and index where layer is being added.
        """
        layer, index = value
        total = self.count() - 1
        widget = QtLayerWidget(layer)
        item = QListWidgetItem(self)
        item.layer = layer
        item.setSizeHint(QSize(228, 32))  # should get height from widget / qss
        self.insertItem(total - index, item)
        self.setItemWidget(item, widget)
        # Block signals to prevent unnecessary selection change calls
        with qt_signals_blocked(self):
            item.setSelected(layer.selected)

    def _on_removed_change(self, value):
        """Remove widget for layer at desired location.

        Parameters
        ----------
        value : 2-tuple
            Tuple of layer and index where layer is being removed.
        """
        _, index = value
        total = self.count() - 1
        item = self.takeItem(total - index)
        del item

    def _on_reordered_change(self, indices):
        """Reorder widgets for layer at desired location.

        Parameters
        ----------
        indices : 2-tuple
            Tuple of old indices and new indices of layers and new indices
        """
        old_indices, new_indices = indices
        total = self.count() - 1

        # Block signals to prevent unnecessary selection change calls
        old_items = [self.item(total - old_index) for old_index in old_indices]
        sorted(new_indices)
        with qt_signals_blocked(self):
            for new_index, old_item in sorted(
                zip(new_indices, old_items), reverse=False
            ):
                widget = self.itemWidget(old_item)
                selected = old_item.isSelected()

                new_item = old_item.clone()
                self.insertItem(total - new_index + 1, new_item)
                self.setItemWidget(new_item, widget)
                new_item.setSelected(selected)

            for old_item in old_items:
                self.takeItem(self.row(old_item))

    def dropEvent(self, event: QEvent):
        """Triggered when the user moves & drops one of the items in the list.

        Parameters
        ----------
        event : QEvent
            The event that triggered the dropEvent.
        """
        event.accept()
        if self.dropIndicatorPosition() == QAbstractItemView.OnViewport:
            return

        total = self.count() - 1
        moving = tuple(total - self.row(item) for item in self.selectedItems())

        insert = self.indexAt(event.pos()).row()
        current = self.currentRow()

        if self.dropIndicatorPosition() == QAbstractItemView.BelowItem:
            insert = insert + 1

        if current == insert or current + 1 == insert:
            return

        if current <= insert:
            insert -= 1

        indices = move_indices(total + 1, moving, total - insert)
        self.events.reordered((indices, tuple(range(total + 1))))

    def startDrag(self, supportedActions: Qt.DropActions):
        drag = drag_with_pixmap(self)
        drag.exec_(supportedActions, Qt.MoveAction)


def move_indices(total, moving, insert):
    index = moving[0]

    # List all indices
    indices = list(range(total))

    # remove all moving to be indices
    for i in moving:
        indices.remove(i)

    # adjust offset based on moving
    offset = sum([i < insert and i != index for i in moving])

    # insert indices to be moved at correct start
    for insert_idx, elem_idx in enumerate(moving, start=insert - offset):
        indices.insert(insert_idx, elem_idx)

    return tuple(indices)
