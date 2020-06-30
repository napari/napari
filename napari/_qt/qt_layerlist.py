from qtpy.QtCore import QSize
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
    centers : list
        List of layer widgets center coordinates.
    layers : napari.components.LayerList
        The layer list to track and display.
    vbox_layout : QVBoxLayout
        The layout instance in which the layouts appear.
    """

    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.layers.events.added.connect(self._add)
        self.layers.events.removed.connect(self._remove)
        # self.layers.events.reordered.connect(self._reorder)
        # self.itemSelectionChanged.connect(self.)

        # Enable drag and drop and widget rearrangement
        self.setSortingEnabled(True)
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip('Layer list')

    def _add(self, event):
        """Insert widget for layer `event.item` at index `event.index`.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        layer = event.item
        total = len(self.layers)
        index = total - event.index
        widget = QtLayerWidget(layer)
        item = QListWidgetItem(self)
        item.setSizeHint(QSize(228, 32))  # should get height from widget
        self.insertItem(index, item)
        self.setItemWidget(item, widget)

    def _remove(self, event):
        """Remove widget for layer at index `event.index`.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        total = len(self.layers)
        index = total - event.index
        item = self.item(index)
        self.removeItemWidget(item)
        del item

    # def _reorder(self, event=None):
    #     """Reorder list of layer widgets.
    #
    #     Loops through all widgets in list, sequentially removing them
    #     and inserting them into the correct place in the final list.
    #
    #     Parameters
    #     ----------
    #     event : qtpy.QtCore.QEvent, optional
    #         Event from the Qt context.
    #     """
    #     total = len(self.layers)
    #
    #     # Create list of layers in order of their widgets
    #     layer_widgets = [self.item(i).widget().layer for i in range(total)]
    #
    #     # Move through the layers in order
    #     for i in range(total):
    #         # Find index of property widget in list of the current layer
    #         index = 2 * indices.index(i)
    #         widget = widgets[index]
    #         divider = widgets[index + 1]
    #         # Check if current index does not match new index
    #         index_current = self.indexOf(widget)
    #         index_new = 2 * (total - i) - 1
    #         if index_current != index_new:
    #             # Remove that property widget and divider
    #             self.removeItemWidget(item)
    #             # Insert the property widget and divider into new location
    #             self.insertItem(index_new, item)

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
