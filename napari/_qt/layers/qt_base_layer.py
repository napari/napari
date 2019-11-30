from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSlider,
    QGridLayout,
    QFrame,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QDoubleSpinBox,
)
import inspect
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


class QtLayerDialog(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.parameters = inspect.signature(self.layer.__init__).parameters

        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.setLayout(self.grid_layout)

        self.nameTextBox = QLineEdit()
        self.nameTextBox.setText(self.layer._basename())
        self.nameTextBox.home(False)
        self.nameTextBox.setToolTip('Layer name')
        self.nameTextBox.setAcceptDrops(False)
        self.nameTextBox.editingFinished.connect(self.changeText)

        self.visibleCheckBox = QCheckBox(self)
        self.visibleCheckBox.setToolTip('Layer visibility')
        self.visibleCheckBox.setChecked(self.parameters['visible'].default)

        self.blendingComboBox = QComboBox()
        for mode in Blending:
            self.blendingComboBox.addItem(str(mode))
        name = self.parameters['blending'].default
        self.blendingComboBox.setCurrentText(str(name))

        self.blendingCheckBox = QCheckBox(self)
        self.blendingCheckBox.setToolTip('Set blending mode')
        self.blendingCheckBox.setChecked(False)
        self.blendingCheckBox.stateChanged.connect(self._on_blending_change)
        self.blendingCheckBox.setChecked(False)
        self._on_blending_change(None)

        self.opacitySpinBox = QDoubleSpinBox()
        self.opacitySpinBox.setToolTip('Opacity')
        self.opacitySpinBox.setKeyboardTracking(False)
        self.opacitySpinBox.setSingleStep(0.01)
        self.opacitySpinBox.setMinimum(0)
        self.opacitySpinBox.setMaximum(1)
        opacity = self.parameters['opacity'].default
        self.opacitySpinBox.setValue(opacity)

        self.scaleTextBox = QLineEdit()
        self.scaleTextBox.setText("")
        self.scaleTextBox.home(False)
        self.scaleTextBox.setToolTip('Layer scale')
        self.scaleTextBox.setAcceptDrops(False)
        self.scaleTextBox.editingFinished.connect(self.change_scale)

        self.translateTextBox = QLineEdit()
        self.translateTextBox.setText("")
        self.translateTextBox.home(False)
        self.translateTextBox.setToolTip('Layer translation')
        self.translateTextBox.setAcceptDrops(False)
        self.translateTextBox.editingFinished.connect(self.change_translate)

        self.metadataTextBox = QLineEdit()
        self.metadataTextBox.setText("")
        self.metadataTextBox.home(False)
        self.metadataTextBox.setToolTip('Layer metadata')
        self.metadataTextBox.setAcceptDrops(False)
        self.metadataTextBox.editingFinished.connect(self.change_metadata)

    def _on_blending_change(self, event):
        state = self.blendingCheckBox.isChecked()
        if state:
            self.blendingComboBox.show()
        else:
            self.blendingComboBox.hide()

    def change_scale(self):
        try:
            scale = eval(self.scaleTextBox.text())
            assert isinstance(scale, list) or isinstance(scale, tuple)
        except (NameError, SyntaxError, AssertionError):
            self.scaleTextBox.setText("")
        self.scaleTextBox.clearFocus()

    def change_translate(self):
        try:
            translate = eval(self.translateTextBox.text())
            assert isinstance(translate, list) or isinstance(translate, tuple)
        except (NameError, SyntaxError, AssertionError):
            self.translateTextBox.setText("")
        self.translateTextBox.clearFocus()

    def change_metadata(self):
        try:
            metadata = eval(self.metadataTextBox.text())
            assert isinstance(metadata, dict)
        except (NameError, SyntaxError, AssertionError):
            self.metadataTextBox.setText("")
        self.metadataTextBox.clearFocus()

    def changeText(self):
        self.nameTextBox.clearFocus()

    def _base_arguments(self):
        """Get keyword arguments for layer creation.

        Returns
        ---------
        arguments : dict
            Keyword arguments for layer creation.
        """
        name = self.nameTextBox.text()
        visible = self.visibleCheckBox.isChecked()

        if self.blendingCheckBox.isChecked():
            blending = self.blendingComboBox.currentText()
        else:
            blending = None

        opacity = self.opacitySpinBox.value()

        scale = self.scaleTextBox.text()
        if scale == "":
            scale = None
        else:
            scale = eval(scale)

        translate = self.translateTextBox.text()
        if translate == "":
            translate = None
        else:
            translate = eval(translate)

        metadata = self.metadataTextBox.text()
        if metadata == "":
            metadata = None
        else:
            metadata = eval(metadata)

        arguments = {
            'name': name,
            'visible': visible,
            'blending': blending,
            'opacity': opacity,
            'scale': scale,
            'translate': translate,
            'metadata': metadata,
        }
        return arguments
