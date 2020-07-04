from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtWidgets import (
    QListWidget,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
)
from .qt_layer_widget import QtLayerWidget
from ..utils.event import Event, EmitterGroup
from .utils import drag_with_pixmap


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
            source=self, auto_connect=False, selection=Event, reordered=Event,
        )

        # When the EVH refactor is fully done we can do the initialization
        # and registering of the listener outside of this class and no longer
        # pass the layers object.
        self.layers = layers
        self.layers.event_handler.register_listener(self)
        self.events.connect(self.layers.event_handler.on_change)

        self.itemSelectionChanged.connect(self._selectionChanged)

        # Enable drag and drop and widget rearrangement
        self.setSortingEnabled(True)
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        # self.setDragDropMode(QAbstractItemView.NoDragDrop)

        # Set selection mode
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.sortItems(Qt.DescendingOrder)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip('Layer list')

        # Once EVH refactor is done, this can be moved to an initialization
        # outside of this object
        self._on_selection_change(self.layers.selected)

    def _selectionChanged(self):
        """Emit an event when selection changes in list widget."""
        total = self.count() - 1
        selected = [total - self.row(item) for item in self.selectedItems()]
        self.events.selection(selected)

    def _on_selection_change(self, selection):
        """When layers selection is changed update the layers list view

        Parmeters
        ---------
        selection : list
            List of selected indices.
        """
        total = self.count() - 1
        for index in range(self.count()):
            item = self.item(index)
            item.setSelected(total - index in selection)

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
        item.setSizeHint(QSize(228, 32))  # should get height from widget / qss
        self.insertItem(total - index, item)
        self.setItemWidget(item, widget)
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

    def _on_reordered_change(self, value):
        """Reorder widgets for layer at desired location.

        Parameters
        ----------
        value : 2-tuple
            Tuple of old indices and new indices of layers and new indices
        """
        old_indices, new_indices = value
        total = self.count() - 1
        print('reordered call', old_indices, new_indices, total)

        widgets = [self.itemWidget(self.item(total - i)) for i in old_indices]
        for index, widget in zip(new_indices, widgets):
            item = self.item(total - index)
            self.removeItemWidget(item)
            print('item widget', item, widget)
            self.setItemWidget(item, widget)

    def dropEvent(self, event: QEvent):
        """Triggered when the user moves & drops one of the items in the list.

        Parameters
        ----------
        event : QEvent
            The event that triggered the dropEvent.
        """
        print('drop call')
        event.accept()
        total = self.count() - 1
        moving = tuple(total - self.row(item) for item in self.selectedItems())
        insert = total - self.indexAt(event.pos()).row()
        indices = move_indices(total + 1, moving, insert)

        print('aaa', indices)
        self.events.reordered((tuple(range(total + 1)), indices))

    def startDrag(self, supportedActions: Qt.DropActions):
        drag = drag_with_pixmap(self)
        drag.exec_(supportedActions, Qt.MoveAction)


def move_indices(total, moving, insert):
    print(total, moving, insert)
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
