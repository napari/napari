from qtpy.QtCore import QSize
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
from qtpy.QtGui import QImage, QPixmap
from .utils import qt_signals_blocked
from ..utils.event import Event, EmitterGroup


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


# TODO: Move this class to layers/qt_layer_widget.py in follow-up refactor
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

        self.events = EmitterGroup(
            source=self,
            auto_connect=False,
            selected=Event,
            name=Event,
            name_unique=Event,
            visible=Event,
        )

        # When the EVH refactor #1376 is done we might not even need the layer
        # attribute anymore as all data updates will be through the handler.
        # At that point we could remove the attribute and do the registering
        # and connecting outside this class and never even need to pass the
        # layer to this class.
        self.layer = layer
        self.layer.event_handler.register_listener(self)
        self.events.connect(self.layer.event_handler.on_change)

        self.setObjectName('layer')

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        tb = QLabel(self)
        tb.setObjectName('thumbnail')
        tb.setToolTip('Layer thumbnail')
        self.thumbnailLabel = tb
        self.layout.addWidget(tb)

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setProperty('mode', 'visibility')
        cb.stateChanged[int].connect(self.events.visible)
        self.visibleCheckBox = cb
        self.layout.addWidget(cb)

        textbox = QLineEdit(self)
        textbox.setAcceptDrops(False)
        textbox.setEnabled(True)
        textbox.editingFinished.connect(self.changeText)
        self.nameTextBox = textbox
        self.nameTextBox.old_name = ''
        self.layout.addWidget(textbox)

        ltb = QLabel(self)
        ltb.setProperty('layer_type_label', True)
        ltb.setToolTip('Layer type')
        self.typeLabel = ltb
        self.layout.addWidget(ltb)

        msg = 'Click to select\nDrag to rearrange'
        self.setToolTip(msg)

        # Once EVH refactor is done, these can be moved to an initialization
        # outside of this object
        self._set_layer_type(type(self.layer).__name__)
        self._on_selected_change(self.layer.selected)
        self._on_thumbnail_change(self.layer.thumbnail)
        self._on_visible_change(self.layer.visible)
        self._on_name_change(self.layer.name)

    def _set_layer_type(self, layer_type):
        """Set layer type.

        Parameters
        ----------
        layer_type : str
            Type of layer, must be one of the napari supported layer types.
        """
        self.typeLabel.setObjectName(layer_type)

    def _on_selected_change(self, selected):
        """Update selected status of the layer widget.

        Parameters
        ----------
        selected : bool
            Layer selected status.
        """
        self.setProperty('selected', selected)
        self.nameTextBox.setEnabled(selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeText(self):
        """Update layer name attribute using layer name textbox contents."""
        name = self.nameTextBox.text()
        old_name = self.nameTextBox.old_name
        if old_name == name:
            return
        self.events.name_unique((old_name, name))
        self.nameTextBox.old_name = self.nameTextBox.text()

        # Prevent retriggering during clearing of focus
        with qt_signals_blocked(self.nameTextBox):
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

    def _on_name_change(self, name):
        """Update text displaying name of layer.

        Parameters
        ----------
        text : str
            Name of the layer.
        """
        self.nameTextBox.setText(name)
        self.nameTextBox.setToolTip(name)
        self.nameTextBox.home(False)
        self.nameTextBox.old_name = name

    def _on_visible_change(self, visible):
        """Toggle visibility of the layer.

        Parameters
        ----------
        visible : bool
            Layer visibility.
        """
        self.visibleCheckBox.setChecked(visible)

    def _on_thumbnail_change(self, thumbnail):
        """Update thumbnail image on the layer widget.

        Parameters
        ----------
        thumbnail : ndarray
            Thumbnail in RGBA unit8 format.
        """
        # Note that QImage expects the image width followed by height
        image = QImage(
            thumbnail,
            thumbnail.shape[1],
            thumbnail.shape[0],
            QImage.Format_RGBA8888,
        )
        self.thumbnailLabel.setPixmap(QPixmap.fromImage(image))
