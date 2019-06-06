from qtpy.QtCore import Qt, QMimeData
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QFrame, QCheckBox, QScrollArea, QApplication)
from qtpy.QtGui import QDrag
import numpy as np


class QtLayersList(QScrollArea):

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
        self.centers = []
        self.setAcceptDrops(True)
        self.setToolTip('Layer list')

        self.layers.events.added.connect(self._add)
        self.layers.events.removed.connect(self._remove)
        self.layers.events.reordered.connect(self._reorder)

        self.drag_start_position = np.zeros(2)
        self.drag_name = None

    def _add(self, event):
        """Insert `event.widget` at index `event.index`."""
        layer = event.item
        index = event.index
        total = len(self.layers)
        if layer._qt_properties is not None:
            self.vbox_layout.insertWidget(2*(total - index)-1,
                                          layer._qt_properties)
            self.vbox_layout.insertWidget(2*(total - index), QtDivider())

    def _remove(self, event):
        """Remove layer widget at index `event.index`."""
        layer = event.item
        if layer._qt_properties is not None:
            index = self.vbox_layout.indexOf(layer._qt_properties)
            divider = self.vbox_layout.itemAt(index+1).widget()
            self.vbox_layout.removeWidget(layer._qt_properties)
            layer._qt_properties.deleteLater()
            layer._qt_properties = None
            self.vbox_layout.removeWidget(divider)
            divider.deleteLater()
            divider = None

    def _reorder(self, event):
        """Reorders list of layer widgets by looping through all
        widgets in list sequentially removing them and inserting
        them into the correct place in final list.
        """
        total = len(self.layers)
        for i in range(total):
            layer = self.layers[i]
            if layer._qt_properties is not None:
                index = self.vbox_layout.indexOf(layer._qt_properties)
                divider = self.vbox_layout.itemAt(index+1).widget()
                self.vbox_layout.removeWidget(layer._qt_properties)
                self.vbox_layout.removeWidget(divider)
                self.vbox_layout.insertWidget(2*(total - i)-1,
                                              layer._qt_properties)
                self.vbox_layout.insertWidget(2*(total - i), divider)

    def mousePressEvent(self, event):
        # Check if mouse press happens on a layer properties widget or
        # a child of such a widget. If not, the press has happended on the
        # Layers Widget itself and should be ignored.
        widget = self.childAt(event.pos())
        layer = (getattr(widget, 'layer', None) or
                 getattr(widget.parentWidget(), 'layer', None))

        if layer is not None:
            self.drag_start_position = np.array([event.pos().x(),
                                                 event.pos().y()])
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
        if (distance < QApplication.startDragDistance() or
            self.drag_name is None):
            return
        mimeData = QMimeData()
        mimeData.setText(self.drag_name)
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos() - self.rect().topLeft())
        dropAction = drag.exec_()

    def dragLeaveEvent(self, event):
        """Unselects layer dividers."""
        event.ignore()
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
        divs = []
        for i in range(0, self.vbox_layout.count(), 2):
            widget = self.vbox_layout.itemAt(i).widget()
            divs.append(widget.y()+widget.frameGeometry().height()/2)
        self.centers = [(divs[i+1]+divs[i])/2 for i in range(len(divs)-1)]
        event.accept()

    def dragMoveEvent(self, event):
        """Set the appropriate layers list divider to be highlighted when
        dragging a layer to a new position in the layers list.
        """
        # Determine which widget center is the mouse currently closed to
        cord = event.pos().y()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        # Determine the current location of the widget being dragged
        total = self.vbox_layout.count()//2 - 1
        insert = total - divider_index
        layer_name = event.mimeData().text()
        if layer_name in self.layers:
            index = self.layers.index(layer_name)
            # If the widget being dragged hasn't moved above or below any other
            # widgets then don't highlight any dividers
            selected = (not (insert == index) and not (insert-1 == index))
            # Set the selected state of all the dividers
            for i in range(0, self.vbox_layout.count(), 2):
                if i == 2*divider_index:
                    self.vbox_layout.itemAt(i).widget().setSelected(selected)
                else:
                    self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dropEvent(self, event):
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        total = self.vbox_layout.count()//2 - 1
        insert = total - divider_index
        layer_name = event.mimeData().text()
        if layer_name in self.layers:
            index = self.layers.index(layer_name)
            if index != insert and index+1 != insert:
                if not self.layers[index].selected:
                    self.layers.unselect_all()
                    self.layers[index].selected = True
                self.layers._move_layers(index, insert)
        event.accept()


class QtDivider(QFrame):
    def __init__(self):
        super().__init__()
        self.setSelected(False)
        self.setFixedSize(50, 2)

    def setSelected(self, selected):
        if selected:
            self.setProperty('selected', True)
            self.style().polish(self)
        else:
            self.setProperty('selected', False)
            self.style().polish(self)
