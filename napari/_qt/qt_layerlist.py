from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QListWidget,
    QLineEdit,
    QLabel,
    QCheckBox,
    QHBoxLayout,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
)


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


class QtLayerWidget(QFrame):
    """Qt view for Layer model.

    Attributes
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    layout : QVBoxLayout
        Layout of the widget.
    nameTextBox : QLineEdit
        Textbox for layer name.
    thumbnailLabel : QLabel
        Label of layer thumbnail.
    typeLabel : QLabel
        Label of layer type.
    visibleCheckBox : QCheckBox
        Checkbox to toggle layer visibility.
    """

    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        layer.events.select.connect(lambda v: self.setSelected(True))
        layer.events.deselect.connect(lambda v: self.setSelected(False))
        layer.events.name.connect(self._on_layer_name_change)
        layer.events.visible.connect(self._on_visible_change)
        layer.events.thumbnail.connect(self._on_thumbnail_change)

        self.setObjectName('layer')

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        tb = QLabel(self)
        tb.setObjectName('thumbnail')
        tb.setToolTip('Layer thumbnail')
        self.thumbnailLabel = tb
        self._on_thumbnail_change()
        self.layout.addWidget(tb)

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.setProperty('mode', 'visibility')
        cb.stateChanged.connect(self.changeVisible)
        self.visibleCheckBox = cb
        self.layout.addWidget(cb)

        textbox = QLineEdit(self)
        textbox.setText(layer.name)
        textbox.home(False)
        textbox.setToolTip(self.layer.name)
        textbox.setAcceptDrops(False)
        textbox.setEnabled(True)
        textbox.editingFinished.connect(self.changeText)
        self.nameTextBox = textbox
        self.layout.addWidget(textbox)

        ltb = QLabel(self)
        layer_type = type(layer).__name__
        ltb.setObjectName(layer_type)
        ltb.setProperty('layer_type_label', True)
        ltb.setToolTip('Layer type')
        self.typeLabel = ltb
        self.layout.addWidget(ltb)

        msg = 'Click to select\nDrag to rearrange'
        self.setToolTip(msg)
        self.setSelected(self.layer.selected)

    def setSelected(self, state):
        """Select layer widget.

        Parameters
        ----------
        state : bool
        """
        self.setProperty('selected', state)
        self.nameTextBox.setEnabled(state)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeVisible(self, state):
        """Toggle visibility of the layer.

        Parameters
        ----------
        state : bool
        """
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def changeText(self):
        """Update layer name attribute using layer name textbox contents."""
        self.layer.name = self.nameTextBox.text()
        self.nameTextBox.setToolTip(self.layer.name)
        self.nameTextBox.clearFocus()
        self.setFocus()

    def mouseReleaseEvent(self, event):
        """Ignores mouse release event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def mousePressEvent(self, event):
        """Ignores mouse press event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def mouseMoveEvent(self, event):
        """Ignores mouse move event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def _on_layer_name_change(self, event=None):
        """Update text displaying name of layer.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional
            Event from the Qt context.
        """
        with self.layer.events.name.blocker():
            self.nameTextBox.setText(self.layer.name)
            self.nameTextBox.home(False)

    def _on_visible_change(self, event=None):
        """Toggle visibility of the layer.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional
            Event from the Qt context.
        """
        with self.layer.events.visible.blocker():
            self.visibleCheckBox.setChecked(self.layer.visible)

    def _on_thumbnail_change(self, event=None):
        """Update thumbnail image on the layer widget.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional
            Event from the Qt context.
        """
        thumbnail = self.layer.thumbnail
        # Note that QImage expects the image width followed by height
        image = QImage(
            thumbnail,
            thumbnail.shape[1],
            thumbnail.shape[0],
            QImage.Format_RGBA8888,
        )
        self.thumbnailLabel.setPixmap(QPixmap.fromImage(image))
