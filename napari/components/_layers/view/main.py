from qtpy.QtCore import Qt, QMimeData
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QFrame, QCheckBox, QScrollArea, QApplication)
from qtpy.QtGui import QDrag


class QtLayers(QScrollArea):

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

        self.dragStartPosition = (0, 0)

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
        widget = self.childAt(event.pos())
        if hasattr(widget, 'layer') or hasattr(widget.parentWidget(), 'layer'):
            self.dragStartPosition = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        """Unselects all layer widgets."""

        widget = self.childAt(event.pos())
        if hasattr(widget, 'layer'):
            layer = widget.layer
        elif hasattr(widget.parentWidget(), 'layer'):
            layer = widget.parentWidget().layer
        else:
            layer = None

        if layer is None:
            self.layers.unselect_all()
        else:
            modifiers = event.modifiers()
            if modifiers == Qt.ShiftModifier:
                index = self.layers.index(layer)
                lastSelected = None
                for i in range(len(self.layers)):
                    if self.layers[i].selected:
                        lastSelected = i
                r = [index, lastSelected]
                r.sort()
                for i in range(r[0], r[1]+1):
                    self.layers[i].selected = True
            elif modifiers == Qt.ControlModifier:
                layer.selected = not layer.selected
            else:
                self.layers.unselect_all(ignore=layer)
                layer.selected = True
        event.accept()

    def mouseMoveEvent(self, event):
        widget = self.childAt(event.pos())
        if hasattr(widget, 'layer'):
            layer = widget.layer
        elif hasattr(widget.parentWidget(), 'layer'):
            layer = widget.parentWidget().layer
        else:
            return

        distance = (event.pos() - self.dragStartPosition).manhattanLength()
        if distance < QApplication.startDragDistance():
            return
        mimeData = QMimeData()
        mimeData.setText(layer.name)
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos() - self.rect().topLeft())
        dropAction = drag.exec_(Qt.MoveAction | Qt.CopyAction)

        if dropAction == Qt.CopyAction:
            if not layer.selected:
                index = self.layers.index(layer)
                self.layers.pop(index)
            else:
                self.layers.remove_selected()
        event.accept()

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
        layer = self.layers[layer_name]
        index = self.layers.index(layer)
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
        layer = self.layers[layer_name]
        index = self.layers.index(layer)
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
