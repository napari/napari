from qtpy.QtCore import Qt, QMimeData, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFrame,
    QScrollArea,
    QApplication,
    QLineEdit,
    QFrame,
    QLabel,
    QCheckBox,
    QHBoxLayout,
    QSizePolicy,
)
from qtpy.QtGui import QDrag
import numpy as np


class QtLayerList(QScrollArea):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
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
        self.dragTimer = QTimer()
        self.dragTimer.setSingleShot(False)
        self.dragTimer.setInterval(20)
        self.dragTimer.timeout.connect(self._force_scroll)
        self._scroll_up = True
        self._min_scroll_region = 24
        self.setAcceptDrops(True)
        self.setToolTip('Layer list')
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.layers.events.added.connect(self._add)
        self.layers.events.removed.connect(self._remove)
        self.layers.events.reordered.connect(lambda e: self._reorder())

        self.drag_start_position = np.zeros(2)
        self.drag_name = None

    def _add(self, event):
        """Insert widget for layer `event.item` at index `event.index`."""
        layer = event.item
        total = len(self.layers)
        index = 2 * (total - event.index) - 1
        widget = QtLayerWidget(layer)
        self.vbox_layout.insertWidget(index, widget)
        self.vbox_layout.insertWidget(index + 1, QtDivider())
        layer.events.select.connect(self._scroll_on_select)

    def _remove(self, event):
        """Remove widget for layer at index `event.index`."""
        layer_index = event.index
        total = len(self.layers)
        # Find property widget and divider for layer to be removed
        index = 2 * (total - layer_index) + 1
        widget = self.vbox_layout.itemAt(index).widget()
        divider = self.vbox_layout.itemAt(index + 1).widget()
        self.vbox_layout.removeWidget(widget)
        widget.deleteLater()
        self.vbox_layout.removeWidget(divider)
        divider.deleteLater()

    def _reorder(self):
        """Reorders list of layer widgets by looping through all
        widgets in list sequentially removing them and inserting
        them into the correct place in final list.
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
        """Scroll to ensure that the currently selected layer is visible."""
        layer = event.source
        self._ensure_visible(layer)

    def _ensure_visible(self, layer):
        """Ensure layer widget for at particular layer is visible."""
        total = len(self.layers)
        layer_index = self.layers.index(layer)
        # Find property widget and divider for layer to be removed
        index = 2 * (total - layer_index) - 1
        widget = self.vbox_layout.itemAt(index).widget()
        self.ensureWidgetVisible(widget)

    def keyPressEvent(self, event):
        event.ignore()

    def keyReleaseEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        # Check if mouse press happens on a layer properties widget or
        # a child of such a widget. If not, the press has happended on the
        # Layers Widget itself and should be ignored.
        widget = self.childAt(event.pos())
        layer = (
            getattr(widget, 'layer', None)
            or getattr(widget.parentWidget(), 'layer', None)
            or getattr(widget.parentWidget().parentWidget(), 'layer', None)
        )

        if layer is not None:
            self.drag_start_position = np.array(
                [event.pos().x(), event.pos().y()]
            )
            self.drag_name = layer.name
        else:
            self.drag_name = None

    def mouseReleaseEvent(self, event):
        if self.drag_name is None:
            # Unselect all the layers if not dragging a layer
            self.layers.unselect_all()
            return

        modifiers = event.modifiers()
        layer = self.layers[self.drag_name]
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
        position = np.array([event.pos().x(), event.pos().y()])
        distance = np.linalg.norm(position - self.drag_start_position)
        if (
            distance < QApplication.startDragDistance()
            or self.drag_name is None
        ):
            return
        mimeData = QMimeData()
        mimeData.setText(self.drag_name)
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos() - self.rect().topLeft())
        dropAction = drag.exec_()
        if self.drag_name is not None:
            index = self.layers.index(self.drag_name)
            layer = self.layers[index]
            self._ensure_visible(layer)

    def dragLeaveEvent(self, event):
        """Unselects layer dividers."""
        event.ignore()
        self.dragTimer.stop()
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
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
        """Set the appropriate layers list divider to be highlighted when
        dragging a layer to a new position in the layers list.
        """
        max_height = self.frameGeometry().height()
        if (
            event.pos().y() < self._min_scroll_region
            and not self.dragTimer.isActive()
        ):
            self._scroll_up = True
            self.dragTimer.start()
        elif (
            event.pos().y() > max_height - self._min_scroll_region
            and not self.dragTimer.isActive()
        ):
            self._scroll_up = False
            self.dragTimer.start()
        elif (
            self.dragTimer.isActive()
            and event.pos().y() >= self._min_scroll_region
            and event.pos().y() <= max_height - self._min_scroll_region
        ):
            self.dragTimer.stop()

        # Determine which widget center is the mouse currently closed to
        cord = event.pos().y() + self.verticalScrollBar().value()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        # Determine the current location of the widget being dragged
        total = self.vbox_layout.count() // 2 - 1
        insert = total - divider_index
        index = self.layers.index(self.drag_name)
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
        if self.dragTimer.isActive():
            self.dragTimer.stop()

        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y() + self.verticalScrollBar().value()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        total = self.vbox_layout.count() // 2 - 1
        insert = total - divider_index
        index = self.layers.index(self.drag_name)
        layer = self.layers[index]
        if index != insert and index + 1 != insert:
            if insert >= index:
                insert -= 1
            self.layers.move_selected(index, insert)
        event.accept()


class QtDivider(QFrame):
    def __init__(self):
        super().__init__()
        self.setSelected(False)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

    def setSelected(self, selected):
        if selected:
            self.setProperty('selected', True)
            self.style().polish(self)
        else:
            self.setProperty('selected', False)
            self.style().polish(self)


class QtLayerWidget(QFrame):
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
        tb.setObjectName('thumbmnail')
        tb.setToolTip('Layer thumbmnail')
        self.thumbnailLabel = tb
        self._on_thumbnail_change(None)
        self.layout.addWidget(tb)

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.setProperty('mode', 'visibility')
        cb.stateChanged.connect(lambda state=cb: self.changeVisible(state))
        self.visibleCheckBox = cb
        self.layout.addWidget(cb)

        textbox = QLineEdit(self)
        textbox.setText(layer.name)
        textbox.home(False)
        textbox.setToolTip('Layer name')
        textbox.setAcceptDrops(False)
        textbox.setEnabled(True)
        textbox.editingFinished.connect(self.changeText)
        self.nameTextBox = textbox
        self.layout.addWidget(textbox)

        ltb = QLabel(self)
        layer_type = type(layer).__name__
        ltb.setObjectName(layer_type)
        ltb.setToolTip('Layer type')
        self.typeLabel = ltb
        self.layout.addWidget(ltb)

        msg = 'Click to select\nDrag to rearrange'
        self.setToolTip(msg)
        self.setSelected(self.layer.selected)

    def setSelected(self, state):
        self.setProperty('selected', state)
        self.nameTextBox.setEnabled(state)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeVisible(self, state):
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def changeText(self):
        self.layer.name = self.nameTextBox.text()
        self.nameTextBox.clearFocus()
        self.setFocus()

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def _on_layer_name_change(self, event):
        with self.layer.events.name.blocker():
            self.nameTextBox.setText(self.layer.name)
            self.nameTextBox.home(False)

    def _on_visible_change(self, event):
        with self.layer.events.visible.blocker():
            self.visibleCheckBox.setChecked(self.layer.visible)

    def _on_thumbnail_change(self, event):
        thumbnail = self.layer.thumbnail
        # Note that QImage expects the image width followed by height
        image = QImage(
            thumbnail,
            thumbnail.shape[1],
            thumbnail.shape[0],
            QImage.Format_RGBA8888,
        )
        self.thumbnailLabel.setPixmap(QPixmap.fromImage(image))
