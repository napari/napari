from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QLabel, QSlider

from ...layers.image._constants import (
    ComplexRendering,
    Interpolation,
    Rendering,
)
from .qt_image_base_layer import QtBaseImageControls


class QtImageControls(QtBaseImageControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.interpolation.connect(self._on_interpolation_change)
        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)
        self.layer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        self.layer.events.data.connect(self._on_data_change)
        self.layer.events.complex_rendering.connect(
            self._on_complex_rendering_change
        )

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

        # complex value combo
        comboBox = QComboBox()
        comboBox.addItems(ComplexRendering.lower_members())
        comboBox.currentTextChanged.connect(self.changeComplex)
        self.complexComboBox = comboBox
        self.complexLabel = QLabel('complex:')

        self._on_ndisplay_change()
        self._on_data_change()

        self.contrastLimitsLabel = QLabel('contrast limits:')
        self.gammaLabel = QLabel('gamma:')

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1, 1, 2)
        self.grid_layout.addWidget(self.contrastLimitsLabel, 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1, 1, 2)
        self.grid_layout.addWidget(self.gammaLabel, 2, 0)
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
        self.grid_layout.addWidget(self.complexLabel, 8, 0)
        self.grid_layout.addWidget(self.complexComboBox, 8, 1, 1, 2)
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

    def changeComplex(self, text):
        # it's possible that a custom function name has beenset
        if text in ComplexRendering.lower_members():
            self.layer.complex_rendering = text

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

    def _on_complex_rendering_change(self, event=None):
        """Set the name of the complex_rendering mode upon change.

        Becuase Image.complex_rendering allows for the user to set custom
        functions, there is extra logic here to update the combo box if an
        unknown function has been set.  We remove them when deselected.
        """
        if isinstance(self.layer.complex_rendering, ComplexRendering):
            text = self.layer.complex_rendering.name.lower()
        else:
            text = self.layer.complex_rendering.__name__.lower()

        if self.layer.complex_rendering == ComplexRendering.COLORMAP:
            self.contrastLimitsLabel.setText('phase limits:')
            self.gammaLabel.setText('mag gamma:')
        else:
            self.contrastLimitsLabel.setText('contrast limits:')
            self.gammaLabel.setText('gamma:')

        # remove any names that may have been added and are no longer valid
        valid = set(ComplexRendering.lower_members() + [text])
        for i in reversed(range(self.complexComboBox.count())):
            if self.complexComboBox.itemText(i) not in valid:
                self.complexComboBox.removeItem(i)
        # if the current option is not in the combo box, add it.
        if self.complexComboBox.findText(text) == -1:
            self.complexComboBox.addItem(text)

        with self.layer.events.complex_rendering.blocker():
            self.complexComboBox.setCurrentText(text)

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

    def _on_data_change(self, event=None):
        if self.layer.is_complex:
            self.complexComboBox.show()
            self.complexLabel.show()
        else:
            self.complexComboBox.hide()
            self.complexLabel.hide()
