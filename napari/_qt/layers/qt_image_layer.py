from qtpy.QtWidgets import QHBoxLayout
from .qt_base_layer import QtLayerControls
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox
from .qt_image_base_layer import QtBaseImageControls
from ...layers.image._constants import Interpolation, Rendering


class QtImageControls(QtBaseImageControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.interpolation.connect(self._on_interpolation_change)
        self.layer.events.rendering.connect(self._on_rendering_change)

        interp_comboBox = QComboBox()
        for interp in Interpolation:
            interp_comboBox.addItem(str(interp))
        index = interp_comboBox.findText(
            self.layer.interpolation, Qt.MatchFixedString
        )
        interp_comboBox.setCurrentIndex(index)
        interp_comboBox.activated[str].connect(
            lambda text=interp_comboBox: self.changeInterpolation(text)
        )
        self.interpComboBox = interp_comboBox
        self.vbox_layout.addWidget(QLabel('interpolation:'))
        self.vbox_layout.addWidget(interp_comboBox)

        renderComboBox = QComboBox()
        for render in Rendering:
            renderComboBox.addItem(str(render))
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.activated[str].connect(
            lambda text=renderComboBox: self.changeRendering(text)
        )
        self.renderComboBox = renderComboBox
        self.vbox_layout.addWidget(QLabel('rendering:'))
        self.vbox_layout.addWidget(renderComboBox)
        self.vbox_layout.addStretch(0)

    def changeInterpolation(self, text):
        self.layer.interpolation = text

    def changeRendering(self, text):
        self.layer.rendering = text

    def _on_interpolation_change(self, event):
        with self.layer.events.interpolation.blocker():
            index = self.interpComboBox.findText(
                self.layer.interpolation, Qt.MatchFixedString
            )
            self.interpComboBox.setCurrentIndex(index)

    def _on_rendering_change(self, event):
        with self.layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self.layer.rendering, Qt.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)
