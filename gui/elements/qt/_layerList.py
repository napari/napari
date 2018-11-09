from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea


class QtLayerList(QScrollArea):
    def __init__(self):
        super().__init__()

        self.setWidgetResizable(True)
        #self.setFixedWidth(315)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollWidget = QWidget()
        self.setWidget(scrollWidget)
        self.layersLayout = QVBoxLayout(scrollWidget)
        self.layersLayout.addStretch(1)

    def add(self, index, layer):
        if layer._qt is not None:
            self.layersLayout.insertWidget(index, layer._qt)

    def remove(self, layer):
        if layer._qt is not None:
            self.layersLayout.removeWidget(layer._qt)
            layer._qt.deleteLater()
            layer._qt = None

    def reorder(self, layerList):
        for i in range(len(layerList)):
            layer = layerList[i]
            if layer._qt is not None:
                self.layersLayout.removeWidget(layer._qt)
                self.layersLayout.insertWidget(i,layer._qt)
