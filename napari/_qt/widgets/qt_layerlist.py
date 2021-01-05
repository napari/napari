from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtCore import QMimeData, QObject, Qt, QTimer
from qtpy.QtGui import QDrag, QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...utils import config
from ...utils.events import disconnect_events

if TYPE_CHECKING:
    from ..experimental.qt_chunk_receiver import QtChunkReceiver


def _create_chunk_receiver(parent: QObject) -> 'Optional[QtChunkReceiver]':
    """Return a QtChunkReceiver or None if not using async.

    The QtChunkReceiver object allows the ChunkLoader to pass newly
    loaded chunks to the layers that requested them.

    Attributes
    ----------
    parent : QObject
        Parent of the chunk receiver.

    Return
    ------
    Optional[QtChunkReceiver]
        The QtChunkReceiver instance to use.
    """
    if config.async_loading:
        from ..experimental.qt_chunk_receiver import QtChunkReceiver

        return QtChunkReceiver(parent)
    else:
        return None


class QtLayerList(QScrollArea):
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
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollWidget = QWidget()
        self.setWidget(scrollWidget)
        self.vbox_layout = QVBoxLayout(scrollWidget)
        self.vbox_layout.addWidget(QtDivider())
        self.vbox_layout.addStretch(1)
        self.vbox_layout.setContentsMargins(0, 0, 0, 0)
        self.vbox_layout.setSpacing(2)
        self.centers = []

        # Create a timer to be used for autoscrolling the layers list up and
        # down when dragging a layer near the end of the displayed area
        self._drag_timer = QTimer()
        self._drag_timer.setSingleShot(False)
        self._drag_timer.setInterval(20)
        self._drag_timer.timeout.connect(self._force_scroll)
        self._scroll_up = True
        self._min_scroll_region = 24
        self.setAcceptDrops(True)
        self.setToolTip('Layer list')
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.layers.events.inserted.connect(self._add)
        self.layers.events.removed.connect(self._remove)
        self.layers.events.reordered.connect(self._reorder)

        self._drag_start_position = np.zeros(2)
        self._drag_name = None

        self.chunk_receiver = _create_chunk_receiver(self)

    def _add(self, event):
        """Insert widget for layer `event.value` at index `event.index`.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        layer = event.value
        total = len(self.layers)
        index = 2 * (total - event.index) - 1
        widget = QtLayerWidget(layer)
        self.vbox_layout.insertWidget(index, widget)
        self.vbox_layout.insertWidget(index + 1, QtDivider())
        layer.events.select.connect(self._scroll_on_select)

    def _remove(self, event):
        """Remove widget for layer at index `event.index`.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        layer_index = event.index
        total = len(self.layers)
        # Find property widget and divider for layer to be removed
        index = 2 * (total - layer_index) + 1
        widget = self.vbox_layout.itemAt(index).widget()
        divider = self.vbox_layout.itemAt(index + 1).widget()
        self.vbox_layout.removeWidget(widget)
        disconnect_events(widget.layer.events, self)
        widget.close()
        self.vbox_layout.removeWidget(divider)
        divider.deleteLater()

    def _reorder(self, event=None):
        """Reorder list of layer widgets.

        Loops through all widgets in list, sequentially removing them
        and inserting them into the correct place in the final list.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        total = len(self.layers)

        # Create list of the current property and divider widgets
        widgets = [
            self.vbox_layout.itemAt(i + 1).widget() for i in range(2 * total)
        ]
        # Take every other widget to ignore the dividers and get just the
        # property widgets
        indices = [
            self.layers.index(w.layer)
            for i, w in enumerate(widgets)
            if i % 2 == 0
        ]

        # Move through the layers in order
        for i in range(total):
            # Find index of property widget in list of the current layer
            index = 2 * indices.index(i)
            widget = widgets[index]
            divider = widgets[index + 1]
            # Check if current index does not match new index
            index_current = self.vbox_layout.indexOf(widget)
            index_new = 2 * (total - i) - 1
            if index_current != index_new:
                # Remove that property widget and divider
                self.vbox_layout.removeWidget(widget)
                self.vbox_layout.removeWidget(divider)
                # Insert the property widget and divider into new location
                self.vbox_layout.insertWidget(index_new, widget)
                self.vbox_layout.insertWidget(index_new + 1, divider)

    def _force_scroll(self):
        """Force the scroll bar to automattically scroll either up or down."""
        cur_value = self.verticalScrollBar().value()
        if self._scroll_up:
            new_value = cur_value - self.verticalScrollBar().singleStep() / 4
            if new_value < 0:
                new_value = 0
            self.verticalScrollBar().setValue(new_value)
        else:
            new_value = cur_value + self.verticalScrollBar().singleStep() / 4
            if new_value > self.verticalScrollBar().maximum():
                new_value = self.verticalScrollBar().maximum()
            self.verticalScrollBar().setValue(new_value)

    def _scroll_on_select(self, event):
        """Scroll to ensure that the currently selected layer is visible.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        layer = event.source
        self._ensure_visible(layer)

    def _ensure_visible(self, layer):
        """Ensure layer widget for at particular layer is visible.

        Parameters
        ----------
        layer : napari.layers.Layer
            An instance of a napari layer.

        """
        total = len(self.layers)
        layer_index = self.layers.index(layer)
        # Find property widget and divider for layer to be removed
        index = 2 * (total - layer_index) - 1
        widget = self.vbox_layout.itemAt(index).widget()
        self.ensureWidgetVisible(widget)

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
        """Ignore key release event.

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
        """Register mouse click if it happens on a layer widget.

        Checks if mouse press happens on a layer properties widget or
        a child of such a widget. If not, the press has happened on the
        Layers Widget itself and should be ignored.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        widget = self.childAt(event.pos())
        layer = (
            getattr(widget, 'layer', None)
            or getattr(widget.parentWidget(), 'layer', None)
            or getattr(widget.parentWidget().parentWidget(), 'layer', None)
        )

        if layer is not None:
            self._drag_start_position = np.array(
                [event.pos().x(), event.pos().y()]
            )
            self._drag_name = layer.name
        else:
            self._drag_name = None

    def mouseReleaseEvent(self, event):
        """Select layer using mouse click.

        Key modifiers:
        Shift - If the Shift button is pressed, select all layers in between
            currently selected one and the clicked one.
        Control - If the Control button is pressed, mouse click will
            toggle selected state of the layer.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        if self._drag_name is None:
            # Unselect all the layers if not dragging a layer
            self.layers.unselect_all()
            return

        modifiers = event.modifiers()
        layer = self.layers[self._drag_name]
        if modifiers == Qt.ShiftModifier:
            # If shift select all layers in between currently selected one and
            # clicked one
            index = self.layers.index(layer)
            lastSelected = None
            for i in range(len(self.layers)):
                if self.layers[i].selected:
                    lastSelected = i
            r = [index, lastSelected]
            r.sort()
            for i in range(r[0], r[1] + 1):
                self.layers[i].selected = True
        elif modifiers == Qt.ControlModifier:
            # If control click toggle selected state
            layer.selected = not layer.selected
        else:
            # If otherwise unselect all and leave clicked one selected
            self.layers.unselect_all(ignore=layer)
            layer.selected = True

    def mouseMoveEvent(self, event):
        """Drag and drop layer with mouse movement.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        position = np.array([event.pos().x(), event.pos().y()])
        distance = np.linalg.norm(position - self._drag_start_position)
        if (
            distance < QApplication.startDragDistance()
            or self._drag_name is None
        ):
            return
        mimeData = QMimeData()
        mimeData.setText(self._drag_name)
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos() - self.rect().topLeft())
        drag.exec_()
        # Check if dragged layer still exists or was deleted during drag
        names = [layer.name for layer in self.layers]
        dragged_layer_exists = self._drag_name in names
        if self._drag_name is not None and dragged_layer_exists:
            index = self.layers.index(self._drag_name)
            layer = self.layers[index]
            self._ensure_visible(layer)

    def dragLeaveEvent(self, event):
        """Unselects layer dividers.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()
        self._drag_timer.stop()
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
        """Update divider position before dragging layer widget to new position

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        if event.source() == self:
            event.accept()
            divs = []
            for i in range(0, self.vbox_layout.count(), 2):
                widget = self.vbox_layout.itemAt(i).widget()
                divs.append(widget.y() + widget.frameGeometry().height() / 2)
            self.centers = [
                (divs[i + 1] + divs[i]) / 2 for i in range(len(divs) - 1)
            ]
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Highlight appriate divider when dragging layer to new position.

        Sets the appropriate layers list divider to be highlighted when
        dragging a layer to a new position in the layers list.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        max_height = self.frameGeometry().height()
        if (
            event.pos().y() < self._min_scroll_region
            and not self._drag_timer.isActive()
        ):
            self._scroll_up = True
            self._drag_timer.start()
        elif (
            event.pos().y() > max_height - self._min_scroll_region
            and not self._drag_timer.isActive()
        ):
            self._scroll_up = False
            self._drag_timer.start()
        elif (
            self._drag_timer.isActive()
            and event.pos().y() >= self._min_scroll_region
            and event.pos().y() <= max_height - self._min_scroll_region
        ):
            self._drag_timer.stop()

        # Determine which widget center is the mouse currently closed to
        cord = event.pos().y() + self.verticalScrollBar().value()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        # Determine the current location of the widget being dragged
        total = self.vbox_layout.count() // 2 - 1
        insert = total - divider_index
        index = self.layers.index(self._drag_name)
        # If the widget being dragged hasn't moved above or below any other
        # widgets then don't highlight any dividers
        selected = not (insert == index) and not (insert - 1 == index)
        # Set the selected state of all the dividers
        for i in range(0, self.vbox_layout.count(), 2):
            if i == 2 * divider_index:
                self.vbox_layout.itemAt(i).widget().setSelected(selected)
            else:
                self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dropEvent(self, event):
        """Drop dragged layer widget into new position in the list of layers.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        if self._drag_timer.isActive():
            self._drag_timer.stop()

        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y() + self.verticalScrollBar().value()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        total = self.vbox_layout.count() // 2 - 1
        insert = total - divider_index
        index = self.layers.index(self._drag_name)
        if index != insert and index + 1 != insert:
            if insert >= index:
                insert -= 1
            self.layers.move_selected(index, insert)
        event.accept()


class QtDivider(QFrame):
    """Qt divider used to separate Qt widgets visually.

    Attributes
    ----------
    layout : QVBoxLayout
        Layout of the widget.
    selected : bool
        Whether the QtDivider is currently selected.
    """

    def __init__(self):
        super().__init__()
        self.setSelected(False)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

    def setSelected(self, selected):
        """Toggle boolean 'selected' attribute of QTDivider instance.

        Parameters
        ----------
        selected : bool
        """
        if selected:
            self.setProperty('selected', True)
            self.style().polish(self)
        else:
            self.setProperty('selected', False)
            self.style().polish(self)


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
        self.layer.events.select.connect(self._on_select_change)
        self.layer.events.deselect.connect(self._on_deselect_change)
        self.layer.events.name.connect(self._on_name_change)
        self.layer.events.visible.connect(self._on_visible_change)
        self.layer.events.thumbnail.connect(self._on_thumbnail_change)

        self.setAttribute(Qt.WA_DeleteOnClose)

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

    def _on_select_change(self, event=None):
        """Update selected state of the layer.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        self.setSelected(True)

    def _on_deselect_change(self, event=None):
        """Update selected state of the layer.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        self.setSelected(False)

    def _on_name_change(self, event=None):
        """Update text displaying name of layer.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.name.blocker():
            self.nameTextBox.setText(self.layer.name)
            self.nameTextBox.home(False)

    def _on_visible_change(self, event=None):
        """Toggle visibility of the layer.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.visible.blocker():
            self.visibleCheckBox.setChecked(self.layer.visible)

    def _on_thumbnail_change(self, event=None):
        """Update thumbnail image on the layer widget.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
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

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        super().close()
