from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from ._layerDivider import QtDivider

class QtLayerList(QScrollArea):
    def __init__(self):
        super().__init__()

        self.setWidgetResizable(True)
        #self.setFixedWidth(315)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollWidget = QWidget()
        self.setWidget(scrollWidget)
        self.layersLayout = QVBoxLayout(scrollWidget)
        self.layersLayout.addWidget(QtDivider())
        self.layersLayout.addStretch(1)
        self.setAcceptDrops(True)

    def insert(self, index, total, layer):
        """Inserts a layer widget at a specific index
        """
        if layer._qt is not None:
            self.layersLayout.insertWidget(2*(total - index)-1, layer._qt)
            self.layersLayout.insertWidget(2*(total - index), QtDivider())

    def remove(self, layer):
        """Removes a layer widget
        """
        if layer._qt is not None:
            index = self.layersLayout.indexOf(layer._qt)
            divider = self.layersLayout.itemAt(index+1).widget()
            self.layersLayout.removeWidget(layer._qt)
            layer._qt.deleteLater()
            layer._qt = None
            self.layersLayout.removeWidget(divider)
            divider.deleteLater()
            divider = None

    def reorder(self, layerList):
        """Reorders list of layer widgets by looping through all
        widgets in list sequentially removing them and inserting
        them into the correct place in final list.
        """
        total = len(layerList)
        for i in range(total):
            layer = layerList[i]
            if layer._qt is not None:
                index = self.layersLayout.indexOf(layer._qt)
                divider = self.layersLayout.itemAt(index+1).widget()
                self.layersLayout.removeWidget(layer._qt)
                self.layersLayout.removeWidget(divider)
                self.layersLayout.insertWidget(2*(total - i)-1,layer._qt)
                self.layersLayout.insertWidget(2*(total - i),divider)

    def mouseReleaseEvent(self, event):
        """Unselects all layer widgets
        """
        self.layersLayout.itemAt(1).widget().unselectAll()

    def dragLeaveEvent(self, event):
        event.ignore()
        for i in range(0, self.layersLayout.count(), 2):
            self.layersLayout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
        event.accept()
        dividers = []
        for i in range(0, self.layersLayout.count(), 2):
            widget = self.layersLayout.itemAt(i).widget()
            dividers.append(widget.y()+widget.frameGeometry().height()/2)
        self.centers = [(dividers[i+1]+dividers[i])/2 for i in range(len(dividers)-1)]

    def dragMoveEvent(self, event):
        cord = event.pos().y()
        divider_index = next((i for i, x in enumerate(self.centers) if x > cord), len(self.centers))
        layerWidget = event.source()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        total = len(layers)
        insert_index = total - divider_index
        if not (insert_index == index) and not (insert_index-1 == index):
            state = True
        else:
            state = False
        for i in range(0, self.layersLayout.count(), 2):
            if i == 2*divider_index:
                self.layersLayout.itemAt(i).widget().setSelected(state)
            else:
                self.layersLayout.itemAt(i).widget().setSelected(False)

    def dropEvent(self, event):
        for i in range(0, self.layersLayout.count(), 2):
            self.layersLayout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y()
        divider_index = next((i for i, x in enumerate(self.centers) if x > cord), len(self.centers))
        layerWidget = event.source()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        total = len(layers)
        insert_index = total - divider_index
        indices = [i for i in range(total)]
        if layerWidget.layer.selected:
            selected = []
            for i in range(total):
                if layers[i].selected:
                    selected.append(i)
        else:
            selected = [index]
        for i in selected:
            indices.remove(i)
        offset = sum([i<insert_index for i in selected])
        j = insert_index - offset
        for i in selected:
            indices.insert(j,i)
            j = j+1
        if not indices == [i for i in range(total)]:
            layers.reorder(indices)
            event.accept()
        else:
            event.ignore()
        if not layerWidget.layer.selected:
            layerWidget.unselectAll()
            layerWidget.setSelected(True)
