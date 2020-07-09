from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import (
    QListWidget,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
)
from .qt_layer_widget import QtLayerWidget
from ..utils.event import Event, EmitterGroup
from ..utils.misc import move_indices
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

    def __init__(self, layers, parent=None):
        super().__init__(parent=parent)

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

        # Set sorting order enabled
        self.setSortingEnabled(True)
        self.sortItems(Qt.DescendingOrder)

        # Set selection mode
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Set sizing
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setSpacing(0)

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
        widget = QtLayerWidget(layer, parent=self)
        item = QListWidgetItem(self)
        # Use the text property of the item for ordering
        item.setText(str(index))
        self.addItem(item)
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
        # Change indices of all items added after item removed from list
        for i in range(index + 1, total + 1):
            item = self.item(total - i)
            item.setText(str(i - 1))

    def _on_reordered_change(self, indices):
        """Reorder widgets for layer at desired location.

        Parameters
        ----------
        indices : 2-tuple
            Tuple of old indices and new indices of layers and new indices
        """
        old_indices, new_indices = indices
        total = self.count() - 1

        old_items = [self.item(total - old_index) for old_index in old_indices]
        # Reorder items by changing their text property
        for new_index, old_item in zip(new_indices, old_items):
            old_item.setText(str(new_index))

    def dropEvent(self, event: QEvent):
        """Triggered when the user moves & drops one of the items in the list.

        Parameters
        ----------
        event : QEvent
            The event that triggered the dropEvent.
        """
        self.setDropIndicatorShown(False)
        event.accept()

        total = self.count() - 1
        moving = tuple(total - self.row(item) for item in self.selectedItems())
        current = self.currentRow()

        if self.dropIndicatorPosition() == QAbstractItemView.OnViewport:
            return
        else:
            insert = self.indexAt(event.pos()).row()

        if self.dropIndicatorPosition() == QAbstractItemView.BelowItem:
            insert = insert + 1

        if current == insert or current + 1 == insert:
            return

        if current <= insert:
            insert -= 1

        indices = move_indices(total + 1, moving[::-1], total - insert)
        self.events.reordered((indices, tuple(range(total + 1))))

    def startDrag(self, supportedActions: Qt.DropActions):
        self.setDropIndicatorShown(True)
        drag = drag_with_pixmap(self, opacity=0.5)
        drag.exec_(supportedActions, Qt.MoveAction)
