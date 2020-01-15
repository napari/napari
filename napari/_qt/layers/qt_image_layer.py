from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox, QSlider
from .qt_image_base_layer import QtBaseImageControls
from ...layers.image._constants import Interpolation, Rendering


class QtImageControls(QtBaseImageControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.interpolation.connect(self._on_interpolation_change)
        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)
        self.layer.dims.events.ndisplay.connect(self._on_ndisplay_change)

        interp_comboBox = QComboBox()
        interp_comboBox.addItems(Interpolation.keys())
        index = interp_comboBox.findText(
            self.layer.interpolation, Qt.MatchFixedString
        )
        interp_comboBox.setCurrentIndex(index)
        interp_comboBox.activated[str].connect(self.changeInterpolation)
        self.interpComboBox = interp_comboBox
        self.interpLabel = QLabel('interpolation:')

        renderComboBox = QComboBox()
        renderComboBox.addItems(Rendering.keys())
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.activated[str].connect(self.changeRendering)
        self.renderComboBox = renderComboBox
        self.renderLabel = QLabel('rendering:')

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.iso_threshold * 100)
        sld.valueChanged.connect(self.changeIsoThreshold)
        self.isoThresholdSlider = sld
        self.isoThresholdLabel = QLabel('iso threshold:')

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(200)
        sld.setSingleStep(1)
        sld.setValue(self.layer.attenuation * 100)
        sld.valueChanged.connect(self.changeAttenuation)
        self.attenuationSlider = sld
        self.attenuationLabel = QLabel('attenuation:')
        self._on_ndisplay_change()

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('gamma:'), 2, 0)
        self.grid_layout.addWidget(self.gammaSlider, 2, 1, 1, 2)
        self.grid_layout.addWidget(self.isoThresholdLabel, 3, 0)
        self.grid_layout.addWidget(self.isoThresholdSlider, 3, 1, 1, 2)
        self.grid_layout.addWidget(self.attenuationLabel, 3, 0)
        self.grid_layout.addWidget(self.attenuationSlider, 3, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('colormap:'), 4, 0)
        self.grid_layout.addWidget(self.colormapComboBox, 4, 2)
        self.grid_layout.addWidget(self.colorbarLabel, 4, 1)
        self.grid_layout.addWidget(QLabel('blending:'), 5, 0)
        self.grid_layout.addWidget(self.blendComboBox, 5, 1, 1, 2)
        self.grid_layout.addWidget(self.renderLabel, 6, 0)
        self.grid_layout.addWidget(self.renderComboBox, 6, 1, 1, 2)
        self.grid_layout.addWidget(self.interpLabel, 7, 0)
        self.grid_layout.addWidget(self.interpComboBox, 7, 1, 1, 2)
        self.grid_layout.setRowStretch(8, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setVerticalSpacing(4)

    def changeInterpolation(self, text):
        self.layer.interpolation = text

    def changeRendering(self, text):
        self.layer.rendering = text
        self._toggle_rendering_parameter_visbility()

    def changeIsoThreshold(self, value):
        with self.layer.events.blocker(self._on_iso_threshold_change):
            self.layer.iso_threshold = value / 100

    def _on_iso_threshold_change(self, event):
        with self.layer.events.iso_threshold.blocker():
            self.isoThresholdSlider.setValue(self.layer.iso_threshold * 100)

    def changeAttenuation(self, value):
        with self.layer.events.blocker(self._on_attenuation_change):
            self.layer.attenuation = value / 100

    def _on_attenuation_change(self, event):
        with self.layer.events.attenuation.blocker():
            self.attenuationSlider.setValue(self.layer.attenuation * 100)

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
            self._toggle_rendering_parameter_visbility()

    def _toggle_rendering_parameter_visbility(self):
        rendering = self.layer.rendering
        if isinstance(rendering, str):
            rendering = Rendering(rendering)
        if rendering == Rendering.ISO:
            self.isoThresholdSlider.show()
            self.isoThresholdLabel.show()
        else:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
        if rendering == Rendering.ATTENUATED_MIP:
            self.attenuationSlider.show()
            self.attenuationLabel.show()
        else:
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()

    def _on_ndisplay_change(self, event=None):
        if self.layer.dims.ndisplay == 2:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()
            self.renderComboBox.hide()
            self.renderLabel.hide()
            self.interpComboBox.show()
            self.interpLabel.show()
        else:
            self.renderComboBox.show()
            self.renderLabel.show()
            self.interpComboBox.hide()
            self.interpLabel.hide()
            self._toggle_rendering_parameter_visbility()
