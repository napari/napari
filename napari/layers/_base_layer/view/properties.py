from qtpy.QtWidgets import (QSlider, QLineEdit, QGridLayout, QFrame, QLabel,
                            QCheckBox, QComboBox)
from qtpy.QtCore import Qt

from .._constants import Blending


class QtLayer(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        layer.events.select.connect(self._on_select)
        layer.events.deselect.connect(self._on_deselect)
        layer.events.name.connect(self._on_layer_name_change)
        layer.events.blending.connect(self._on_blending_change)
        layer.events.opacity.connect(self._on_opacity_change)
        layer.events.visible.connect(self._on_visible_change)

        self.setObjectName('layer')

        self.grid_layout = QGridLayout()

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.setProperty('mode', 'visibility')
        cb.stateChanged.connect(lambda state=cb: self.changeVisible(state))
        self.visibleCheckBox = cb
        self.grid_layout.addWidget(cb, 0, 0)

        textbox = QLineEdit(self)
        textbox.setText(layer.name)
        textbox.home(False)
        textbox.setToolTip('Layer name')
        textbox.setFixedWidth(122)
        textbox.setAcceptDrops(False)
        textbox.setEnabled(True)
        textbox.editingFinished.connect(self.changeText)
        self.nameTextBox = textbox
        self.grid_layout.addWidget(textbox, 0, 1)

        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0)
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(110)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.opacity*100)
        sld.valueChanged[int].connect(
            lambda value=sld: self.changeOpacity(value))
        self.opacitySilder = sld
        self.grid_layout.addWidget(sld, 1, 1)

        blend_comboBox = QComboBox()
        for blend in Blending:
            blend_comboBox.addItem(str(blend))
        index = blend_comboBox.findText(
            self.layer.blending, Qt.MatchFixedString)
        blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(
            lambda text=blend_comboBox: self.changeBlending(text))
        self.blendComboBox = blend_comboBox
        self.grid_layout.addWidget(QLabel('blending:'), 2, 0)
        self.grid_layout.addWidget(blend_comboBox, 2, 1)

        self.setLayout(self.grid_layout)
        msg = 'Click to select\nDrag to rearrange\nDouble click to expand'
        self.setToolTip(msg)
        self.setExpanded(False)
        self.setFixedWidth(250)
        self.grid_layout.setColumnMinimumWidth(0, 100)
        self.grid_layout.setColumnMinimumWidth(1, 100)
        self.layer.selected = True

    def _on_select(self, event):
        self.setProperty('selected', True)
        self.nameTextBox.setEnabled(True)
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_deselect(self, event):
        self.setProperty('selected', False)
        self.nameTextBox.setEnabled(False)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeOpacity(self, value):
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value/100

    def changeVisible(self, state):
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def changeText(self):
        self.layer.name = self.nameTextBox.text()

    def changeBlending(self, text):
        self.layer.blending = text

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        self.setExpanded(not self.expanded)

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

    def _on_layer_name_change(self, event):
        with self.layer.events.name.blocker():
            self.nameTextBox.setText(self.layer.name)
            self.nameTextBox.home(False)

    def _on_opacity_change(self, event):
        with self.layer.events.opacity.blocker():
            self.opacitySilder.setValue(self.layer.opacity*100)

    def _on_blending_change(self, event):
        with self.layer.events.blending.blocker():
            index = self.blendComboBox.findText(
                self.layer.blending, Qt.MatchFixedString)
            self.blendComboBox.setCurrentIndex(index)

    def _on_visible_change(self, event):
        with self.layer.events.visible.blocker():
            self.visibleCheckBox.setChecked(self.layer.visible)
