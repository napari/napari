from qtpy.QtCore import QSize, QModelIndex
from qtpy.QtWidgets import (
    QListWidget,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
)
from .qt_layer_widget import QtLayerWidget
from ..utils.event import Event, EmitterGroup


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
            source=self, auto_connect=False, selection=Event,
        )

        # When the EVH refactor is fully done we can do the initialization
        # and registering of the listener outside of this class and no longer
        # pass the layers object.
        self.layers = layers
        self.layers.event_handler.register_listener(self)
        self.events.connect(self.layers.event_handler.on_change)

        self.model().rowsMoved[
            QModelIndex, int, int, QModelIndex, int
        ].connect(self._reorder)
        self.itemSelectionChanged.connect(self._selectionChanged)

        # Enable drag and drop and widget rearrangement
        self.setSortingEnabled(True)
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        # Set selection mode
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip('Layer list')

        # Once EVH refactor is done, this can be moved to an initialization
        # outside of this object
        selection = [
            i for i, layer in enumerate(self.layers) if layer.selected
        ]
        print(selection)
        self._on_selection_change(selection)

    def _reorder(self, parent, start, end, destination, row):
        print(start, end, row)

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
        print('asdfasdfsaf', selection, self.count())
        total = self.count() - 1
        for index in range(self.count()):
            item = self.item(index)
            print('rrrr', total - index, selection)
            item.setSelected(total - index in selection)

    def _on_added_change(self, value):
        """Insert widget for layer at desired location.

        Parameters
        ----------
        value : 2-tuple
            Tuple of layer and index where layer is being added.
        """
        print('qqqqqqq', value[1])
        layer, index = value
        total = self.count() - 1
        widget = QtLayerWidget(layer)
        item = QListWidgetItem(self)
        item.setSizeHint(QSize(228, 32))  # should get height from widget / qss
        self.insertItem(total - index, item)
        self.setItemWidget(item, widget)
        print('asdf', layer.selected)
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
        item = self.item(total - index)
        self.removeItemWidget(item)
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

        for old_index, new_index in zip(old_indices, new_indices):
            item = self.takeItem(total - old_index)
            self.insertItem(total - new_index, item)
