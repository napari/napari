from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox, QCheckBox
from .qt_image_base_layer import QtBaseImageControls, QtBaseImageDialog
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

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0, 1, 3)
        self.grid_layout.addWidget(self.opacitySilder, 0, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0, 1, 3)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('gamma:'), 2, 0, 1, 3)
        self.grid_layout.addWidget(self.gammaSlider, 2, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('colormap:'), 3, 0, 1, 3)
        self.grid_layout.addWidget(self.colormapComboBox, 3, 3, 1, 3)
        self.grid_layout.addWidget(self.colorbarLabel, 3, 6)
        self.grid_layout.addWidget(QLabel('blending:'), 4, 0, 1, 3)
        self.grid_layout.addWidget(self.blendComboBox, 4, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('rendering:'), 5, 0, 1, 3)
        self.grid_layout.addWidget(self.renderComboBox, 5, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('interpolation:'), 6, 0, 1, 3)
        self.grid_layout.addWidget(self.interpComboBox, 6, 3, 1, 4)
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


class QtImageDialog(QtBaseImageDialog):
    def __init__(self, layer):
        super().__init__(layer)

        self.pyramidCheckBox = QCheckBox(self)
        self.pyramidCheckBox.setToolTip('Is pyramid')
        # self.pyramidCheckBox.setChecked(self.parameters['is_pyramid'].default)

        self.rgbCheckBox = QCheckBox(self)
        self.rgbCheckBox.setToolTip('Is rgb')
        # self.rgbCheckBox.setChecked(self.parameters['rgb'].default)

        self.grid_layout.addWidget(QLabel('name:'), 0, 0)
        self.grid_layout.addWidget(self.nameTextBox, 0, 1)
        self.grid_layout.addWidget(QLabel('visible:'), 1, 0)
        self.grid_layout.addWidget(self.visibleCheckBox, 1, 1)
        self.grid_layout.addWidget(QLabel('is_pyramid:'), 2, 0)
        self.grid_layout.addWidget(self.pyramidCheckBox, 2, 1)
        self.grid_layout.addWidget(QLabel('rgb:'), 3, 0)
        self.grid_layout.addWidget(self.rgbCheckBox, 3, 1)

    def get_arguments(self):
        """Get keyword arguments for layer creation.

        Returns
        ---------
        arguments : dict
            Keyword arguments for layer creation.
        """
        base_arguments = self._base_arguments()
        is_pyramid = self.pyramidCheckBox.isChecked()
        rgb = self.rgbCheckBox.isChecked()

        arguments = {'is_pyramid': is_pyramid, 'rgb': rgb}
        arguments.update(base_arguments)

        return arguments
