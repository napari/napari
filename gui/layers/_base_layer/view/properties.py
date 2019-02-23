from PyQt5.QtWidgets import (QSlider, QLineEdit, QGridLayout, QFrame,
                             QVBoxLayout, QCheckBox, QWidget, QApplication,
                             QLabel, QComboBox)
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QPalette, QDrag

from os.path import join
from ....resources import resources_dir
path_on = join(resources_dir, 'icons', 'eye_on.png')


class QtLayer(QFrame):
    unselectedStylesheet = """QFrame#layer {border: 3px solid lightGray;
        background-color:lightGray; border-radius: 3px;}"""

    selectedStylesheet = """QFrame#layer {border: 3px solid rgb(0, 153, 255);
        background-color:lightGray; border-radius: 3px;}"""

    cbStylesheet = """QCheckBox::indicator {width: 18px; height: 18px;}
        QCheckBox::indicator:checked {image: url(""" + path_on + ");}"

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layer.events.select.connect(self._on_select)
        self.layer.events.deselect.connect(self._on_deselect)

        self.setObjectName('layer')
        self.layer.selected = True

        self.grid_layout = QGridLayout()

        cb = QCheckBox(self)
        cb.setStyleSheet(self.cbStylesheet)
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.stateChanged.connect(lambda state=cb: self.changeVisible(state))
        self.grid_layout.addWidget(cb, 0, 0)

        textbox = QLineEdit(self)
        textbox.setStyleSheet('background-color:lightGray; border:none')
        textbox.setText(layer.name)
        textbox.setToolTip('Layer name')
        textbox.setFixedWidth(80)
        textbox.setAcceptDrops(False)
        textbox.editingFinished.connect(
            lambda text=textbox: self.changeText(text))
        self.grid_layout.addWidget(textbox, 0, 1)

        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0)
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.opacity*100)
        sld.valueChanged[int].connect(
            lambda value=sld: self.changeOpacity(value))
        self.grid_layout.addWidget(sld, 1, 1)

        blend_comboBox = QComboBox()
        for blend in self.layer._blending_modes:
            blend_comboBox.addItem(blend)
        index = blend_comboBox.findText(
            self.layer._blending, Qt.MatchFixedString)
        if index >= 0:
            blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(
            lambda text=blend_comboBox: self.changeBlending(text))
        self.grid_layout.addWidget(QLabel('blending:'), 2, 0)
        self.grid_layout.addWidget(blend_comboBox, 2, 1)

        self.setLayout(self.grid_layout)
        msg = 'Click to select\nDrag to rearrange\nDouble click to expand'
        self.setToolTip(msg)
        self.setExpanded(False)
        self.setFixedWidth(200)
        self.grid_layout.setColumnMinimumWidth(0, 100)
        self.grid_layout.setColumnMinimumWidth(1, 100)

    def _on_select(self, event):
        self.setStyleSheet(self.selectedStylesheet)

    def _on_deselect(self, event):
        self.setStyleSheet(self.unselectedStylesheet)

    def changeOpacity(self, value):
        self.layer.opacity = value/100

    def changeVisible(self, state):
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def changeText(self, text):
        self.layer.name = text.text()

    def changeBlending(self, text):
        self.layer.blending = text

    def mouseReleaseEvent(self, event):
        modifiers = event.modifiers()
        if modifiers == Qt.ShiftModifier:
            index = self.layer.viewer.layers.index(self.layer)
            lastSelected = None
            for i in range(len(self.layer.viewer.layers)):
                if self.layer.viewer.layers[i].selected:
                    lastSelected = i
            r = [index, lastSelected]
            r.sort()
            for i in range(r[0], r[1]+1):
                self.layer.viewer.layers[i].selected = True
        elif modifiers == Qt.ControlModifier:
            self.layer.selected = not self.layer.selected
        else:
            self.layer.viewer.layers.unselect_all()
            self.layer.selected = True

    def mousePressEvent(self, event):
        self.dragStartPosition = event.pos()

    def mouseMoveEvent(self, event):
        distance = (event.pos() - self.dragStartPosition).manhattanLength()
        if distance < QApplication.startDragDistance():
            return
        mimeData = QMimeData()
        if not self.layer.selected:
            name = self.layer.name
        else:
            name = ''
            for layer in self.layer.viewer.layers:
                if layer.selected:
                    name = layer.name + '; ' + name
            name = name[:-2]
        mimeData.setText(name)
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos() - self.rect().topLeft())
        dropAction = drag.exec_(Qt.MoveAction | Qt.CopyAction)

        if dropAction == Qt.CopyAction:
            if not self.layer.selected:
                index = self.layer.viewer.layers.index(self.layer)
                self.layer.viewer.layers.pop(index)
            else:
                self.layer.viewer.layers.remove_selected()

    def setExpanded(self, bool):
        if bool:
            self.expanded = True
            rows = self.grid_layout.rowCount()
            self.setFixedHeight(55*(rows-1))
        else:
            self.expanded = False
            self.setFixedHeight(55)
        rows = self.grid_layout.rowCount()
        for i in range(1, rows):
            for j in range(2):
                if self.expanded:
                    self.grid_layout.itemAtPosition(i, j).widget().show()
                else:
                    self.grid_layout.itemAtPosition(i, j).widget().hide()

    def mouseDoubleClickEvent(self, event):
        self.setExpanded(not self.expanded)

    def update(self):
        return
