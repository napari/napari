from qtpy.QtCore import QSize, QModelIndex
from qtpy.QtWidgets import (
    QListWidget,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
)
from .qt_layer_widget import QtLayerWidget


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

        # When the EVH refactor is fully done we can do the initialization
        # and registering of the listener outside of this class and no longer
        # pass the layers object.
        self.layers = layers
        self.layers.event_handler.register_listener(self)

        self.model().rowsMoved[
            QModelIndex, int, int, QModelIndex, int
        ].connect(self._reorder)
        # self.itemSelectionChanged.connect(self.)

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

    def _reorder(self, parent, start, end, destination, row):
        print(start, end, row)

    def _on_added_change(self, value):
        """Insert widget for layer at desired location.

        Parameters
        ----------
        value : 2-tuple
            Tuple of layer and index where layer is being added.
        """
        layer, index = value
        total = self.count()
        widget = QtLayerWidget(layer)
        item = QListWidgetItem(self)
        item.setSizeHint(QSize(228, 32))  # should get height from widget / qss
        self.insertItem(total - index, item)
        self.setItemWidget(item, widget)

    def _on_removed_change(self, value):
        """Remove widget for layer at desired location.

        Parameters
        ----------
        value : 2-tuple
            Tuple of layer and index where layer is being removed.
        """
        _, index = value
        total = self.count()
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
        total = self.count()

        for old_index, new_index in zip(old_indices, new_indices):
            item = self.takeItem(total - old_index)
            self.insertItem(total - new_index, item)

    def keyPressEvent(self, event):
        """Ignore a key press event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def keyReleaseEvent(self, event):
        """Ignore key relase event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()
