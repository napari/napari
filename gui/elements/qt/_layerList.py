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
        self.layersLayout.addWidget(QtDivider('TOP'))
        self.layersLayout.addStretch(1)

    def insert(self, index, total, layer):
        """Inserts a layer widget at a specific index
        """
        if layer._qt is not None:
            self.layersLayout.insertWidget(2*(total - index)-1, layer._qt)
            self.layersLayout.insertWidget(2*(total - index), QtDivider('ADD'))

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
        for i in range(len(layerList)):
            layer = layerList[i]
            if layer._qt is not None:
                self.layersLayout.removeWidget(layer._qt)
                self.layersLayout.insertWidget(2*(len(layerList) - i)-1,layer._qt)

    def mouseReleaseEvent(self, event):
        """Unselects all layer widgets
        """
        self.layersLayout.itemAt(1).widget().unselectAll()
