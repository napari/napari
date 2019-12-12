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
        self.layer.dims.events.ndisplay.connect(self._on_ndisplay_change)

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
        self.interpLabel = QLabel('interpolation:')

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
        self.renderLabel = QLabel('rendering:')

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.iso_threshold * 100)
        sld.valueChanged[int].connect(
            lambda value=sld: self.changeIsoTheshold(value)
        )
        self.isoThesholdSilder = sld
        self.isoThesholdLabel = QLabel('iso threshold:')
        self._on_ndisplay_change(None)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0)
        self.grid_layout.addWidget(self.opacitySilder, 0, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('gamma:'), 2, 0)
        self.grid_layout.addWidget(self.gammaSlider, 2, 1, 1, 2)
        self.grid_layout.addWidget(self.isoThesholdLabel, 3, 0)
        self.grid_layout.addWidget(self.isoThesholdSilder, 3, 1, 1, 2)
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
        self._toggle_iso_threhold_visbility()

    def changeIsoTheshold(self, value):
        with self.layer.events.blocker(self._on_iso_threshold_change):
            self.layer.iso_threshold = value / 100

    def _on_iso_threshold_change(self, event):
        with self.layer.events.iso_threshold.blocker():
            self.isoThesholdSilder.setValue(self.layer.iso_threshold * 100)

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
            self._toggle_iso_threhold_visbility()

    def _toggle_iso_threhold_visbility(self):
        rendering = self.layer.rendering
        if isinstance(rendering, str):
            rendering = Rendering(rendering)
        if rendering == Rendering.ISO:
            self.isoThesholdSilder.show()
            self.isoThesholdLabel.show()
        else:
            self.isoThesholdSilder.hide()
            self.isoThesholdLabel.hide()

    def _on_ndisplay_change(self, event):
        if self.layer.dims.ndisplay == 2:
            self.isoThesholdSilder.hide()
            self.isoThesholdLabel.hide()
            self.renderComboBox.hide()
            self.renderLabel.hide()
            self.interpComboBox.show()
            self.interpLabel.show()
        else:
            self.renderComboBox.show()
            self.renderLabel.show()
            self.interpComboBox.hide()
            self.interpLabel.hide()
            self._toggle_iso_threhold_visbility()
