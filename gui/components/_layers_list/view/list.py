from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QCheckBox, QScrollArea)


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
        self.centers = []
        self.setAcceptDrops(True)
        self.setToolTip('Layer list')

        self.layers.changed.added.connect(self._add)
        self.layers.changed.removed.connect(self._remove)
        self.layers.changed.reordered.connect(self._reorder)

    def _add(self, event):
        """Inserts a layer widget at a specific index
        """
        layer = event.item
        index = event.index
        total = len(self.layers)
        if layer._qt_properties is not None:
            self.vbox_layout.insertWidget(2*(total - index)-1,
                                          layer._qt_properties)
            self.vbox_layout.insertWidget(2*(total - index), QtDivider())

    def _remove(self, event):
        """Removes a layer widget
        """
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

    def mouseReleaseEvent(self, event):
        """Unselects all layer widgets
        """
        self.layers.unselect_all()

    def dragLeaveEvent(self, event):
        """Unselects layer dividers
        """
        event.ignore()
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
        event.accept()
        divs = []
        for i in range(0, self.vbox_layout.count(), 2):
            widget = self.vbox_layout.itemAt(i).widget()
            divs.append(widget.y()+widget.frameGeometry().height()/2)
        self.centers = [(divs[i+1]+divs[i])/2 for i in range(len(divs)-1)]

    def dragMoveEvent(self, event):
        cord = event.pos().y()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        layerWidget = event.source()
        total = self.vbox_layout.count()//2 - 1
        index = total - self.vbox_layout.indexOf(layerWidget)//2 - 1
        insert = total - divider_index
        if not (insert == index) and not (insert-1 == index):
            state = True
        else:
            state = False
        for i in range(0, self.vbox_layout.count(), 2):
            if i == 2*divider_index:
                self.vbox_layout.itemAt(i).widget().setSelected(state)
            else:
                self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dropEvent(self, event):
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y()
        center_list = (i for i, x in enumerate(self.centers) if x > cord)
        divider_index = next(center_list, len(self.centers))
        layerWidget = event.source()
        total = self.vbox_layout.count()//2 - 1
        index = total - self.vbox_layout.indexOf(layerWidget)//2 - 1
        insert = total - divider_index
        if index != insert and index+1 != insert:
            if not self.layers[index].selected:
                self.layers.unselect_all()
                self.layers[index].selected = True
            self.layers._move_layers(index, insert)
        event.accept()


class QtDivider(QFrame):
    unselectedStlyesheet = """QFrame {border: 3px solid rgb(236,236,236);
        background-color:rgb(236,236,236); border-radius: 3px;}"""
    selectedStlyesheet = """QFrame {border: 3px solid rgb(0, 153, 255);
        background-color:rgb(0, 153, 255); border-radius: 3px;}"""

    def __init__(self):
        super().__init__()
        self.setSelected(False)
        self.setFixedHeight(4)

    def setSelected(self, bool):
        if bool:
            self.setStyleSheet(self.selectedStlyesheet)
        else:
            self.setStyleSheet(self.unselectedStlyesheet)
