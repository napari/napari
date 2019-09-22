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

        layout_option = 2
        if layout_option == 1:
            self.grid_layout.addWidget(QLabel('opacity:'), 0, 0, 1, 3)
            self.grid_layout.addWidget(self.opacitySilder, 1, 0, 1, 3)
            self.grid_layout.addWidget(QLabel('contrast limits:'), 2, 0, 1, 3)
            self.grid_layout.addWidget(self.contrastLimitsSlider, 3, 0, 1, 3)
            self.grid_layout.addWidget(QLabel('colormap:'), 4, 0, 1, 3)
            self.grid_layout.addWidget(self.colormapComboBox, 5, 0, 1, 3)
            self.grid_layout.addWidget(QLabel('blending:'), 6, 0, 1, 3)
            self.grid_layout.addWidget(self.blendComboBox, 7, 0, 1, 3)
            self.grid_layout.addWidget(QLabel('rendering:'), 8, 0, 1, 3)
            self.grid_layout.addWidget(self.renderComboBox, 9, 0, 1, 3)
            self.grid_layout.addWidget(QLabel('interpolation:'), 10, 0, 1, 3)
            self.grid_layout.addWidget(self.interpComboBox, 11, 0, 1, 3)
            self.grid_layout.setRowStretch(12, 1)
        elif layout_option == 2:
            self.grid_layout.addWidget(QLabel('opacity:'), 0, 0, 1, 3)
            self.grid_layout.addWidget(self.opacitySilder, 0, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0, 1, 3)
            self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('colormap:'), 2, 0, 1, 3)
            self.grid_layout.addWidget(self.colormapComboBox, 2, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('blending:'), 3, 0, 1, 3)
            self.grid_layout.addWidget(self.blendComboBox, 3, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('rendering:'), 4, 0, 1, 3)
            self.grid_layout.addWidget(self.renderComboBox, 4, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('interpolation:'), 5, 0, 1, 3)
            self.grid_layout.addWidget(self.interpComboBox, 5, 3, 1, 4)
            self.grid_layout.setRowStretch(6, 1)
            self.grid_layout.setVerticalSpacing(4)

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
