from PyQt5.QtCore import Qt
from ._layer import QtLayer

class QtImageLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        #comboBox = self.children()[3]
        #for cmap in self.layer.colormaps:
        #    comboBox.addItem(cmap)
        #index = comboBox.findText('hot', Qt.MatchFixedString)
        #if index >= 0:
        #    comboBox.setCurrentIndex(index)
        #comboBox.activated[str].connect(lambda text=comboBox: self.changeColor(text))

    def changeColor(self, text):
        self.layer.colormap = text
