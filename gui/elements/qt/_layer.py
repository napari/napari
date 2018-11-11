from PyQt5.QtWidgets import QSlider, QLineEdit, QHBoxLayout, QGroupBox, QVBoxLayout, QCheckBox, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette

import weakref

class QtLayer(QGroupBox):
    def __init__(self, layer):
        super().__init__()
        self.layer = weakref.proxy(layer)
        layout = QHBoxLayout()

        cb = QCheckBox(self)
        cb.setStyleSheet("QCheckBox::indicator {width: 18px; height: 18px;}"
                         "QCheckBox::indicator:unchecked {image: url(eye_off.png);}"
                         "QCheckBox::indicator:checked {image: url(eye_on.png);}")
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.stateChanged.connect(lambda state=cb: self.changeVisible(state))
        layout.addWidget(cb)

        textbox = QLineEdit(self)
        textbox.setStyleSheet('background-color:lightGray; border:none')
        textbox.setText(layer.name)
        textbox.setToolTip('Layer name')
        textbox.setFixedWidth(80)
        layout.addWidget(textbox)

        sld = QSlider(Qt.Horizontal, self)
        sld.setToolTip('Layer opacity')
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.opacity*100)
        sld.valueChanged[int].connect(lambda value=sld: self.changeOpacity(value))
        layout.addWidget(sld)

        layout.insertSpacing(1, 5)

        self.setLayout(layout)
        self.setFixedHeight(55)
        self.setStyleSheet("QGroupBox {border: 3px solid lightGray; background-color:lightGray;"
            "border-radius: 3px;}")

    def changeOpacity(self, value):
        self.layer.opacity = value/100

    def changeVisible(self, state):
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def mouseReleaseEvent(self, event):
        if event.modifiers == Qt.ShiftModifier:
            print('shift!!!')
        else:
            print('no shift!!!')
        self.layer.selected = True
        self.setStyleSheet("QGroupBox { border: 3px solid darkBlue; background-color:lightGray;"
            "border-radius: 3px;}")

    def update(self):
        print('hello!!!')
        #print(self.layout().children())
        #sld.setValue(self.layer.opacity*100)
        #cb.setChecked(self.layer.visible)
