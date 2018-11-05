from PyQt5.QtWidgets import QSlider, QComboBox, QHBoxLayout, QGroupBox, QVBoxLayout, QCheckBox
from PyQt5.QtCore import Qt

class QtLayer(QGroupBox):
    def __init__(self, layer):
        super().__init__(layer.name)
        self.layer = layer
        layout = QHBoxLayout()

        cb = QCheckBox('Visibility', self)
        cb.setChecked(self.layer.visible)
        cb.stateChanged.connect(lambda state=cb: self.changeVisible(state))
        layout.addWidget(cb)

        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        index = comboBox.findText('hot', Qt.MatchFixedString)
        if index >= 0:
            comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(lambda text=comboBox: self.onActivated(text))
        layout.addWidget(comboBox)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.opacity*100)
        sld.valueChanged[int].connect(lambda value=sld: self.changeValue(value))
        layout.addWidget(sld)

        self.setLayout(layout)
        self.setFixedHeight(75)

    def changeValue(self, value):
        self.layer.opacity = value/100

    def changeVisible(self, state):
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def onActivated(self, text):
        self.layer.colormap = text

    def update(self):
        print('hello!!!')
        #print(self.layout().children())
        #sld.setValue(self.layer.opacity*100)
        #cb.setChecked(self.layer.visible)
