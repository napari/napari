from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QSlider,
    QLineEdit,
    QGridLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QPushButton,
)

from ...layers.base._constants import Blending


class QtLayerControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        layer.events.blending.connect(self._on_blending_change)
        layer.events.opacity.connect(self._on_opacity_change)
        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.setLayout(self.grid_layout)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.opacity * 100)
        sld.valueChanged[int].connect(
            lambda value=sld: self.changeOpacity(value)
        )
        self.opacitySilder = sld

        blend_comboBox = QComboBox()
        for blend in Blending:
            blend_comboBox.addItem(str(blend))
        index = blend_comboBox.findText(
            self.layer.blending, Qt.MatchFixedString
        )
        blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(
            lambda text=blend_comboBox: self.changeBlending(text)
        )
        self.blendComboBox = blend_comboBox

    def changeOpacity(self, value):
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value / 100

    def changeBlending(self, text):
        self.layer.blending = text

    def _on_opacity_change(self, event):
        with self.layer.events.opacity.blocker():
            self.opacitySilder.setValue(self.layer.opacity * 100)

    def _on_blending_change(self, event):
        with self.layer.events.blending.blocker():
            index = self.blendComboBox.findText(
                self.layer.blending, Qt.MatchFixedString
            )
            self.blendComboBox.setCurrentIndex(index)
