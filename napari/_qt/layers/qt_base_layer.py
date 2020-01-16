from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider, QGridLayout, QFrame, QComboBox

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
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self._on_opacity_change()

        blend_comboBox = QComboBox()
        blend_comboBox.addItems(Blending.keys())
        index = blend_comboBox.findText(
            self.layer.blending, Qt.MatchFixedString
        )
        blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(self.changeBlending)
        self.blendComboBox = blend_comboBox

    def changeOpacity(self, value):
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value / 100

    def changeBlending(self, text):
        self.layer.blending = text

    def _on_opacity_change(self, event=None):
        with self.layer.events.opacity.blocker():
            self.opacitySlider.setValue(self.layer.opacity * 100)

    def _on_blending_change(self, event=None):
        with self.layer.events.blending.blocker():
            index = self.blendComboBox.findText(
                self.layer.blending, Qt.MatchFixedString
            )
            self.blendComboBox.setCurrentIndex(index)
