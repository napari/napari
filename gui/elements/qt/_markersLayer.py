from PyQt5.QtCore import Qt
from ._layer import QtLayer

class QtMarkersLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        #comboBox = self.children()[3]
        #colors = ['red', 'blue', 'green', 'white']
        #for c in colors:
        #    comboBox.addItem(c)
        #index = comboBox.findText('white', Qt.MatchFixedString)
        #if index >= 0:
        #    comboBox.setCurrentIndex(index)
        #comboBox.activated[str].connect(lambda text=comboBox: self.changeColor(text))

    def changeColor(self, text):
        self.layer.face_color = text
