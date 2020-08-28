import numpy as np
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels

np.random.seed(0)
_LABELS = np.random.randint(20, size=(10, 15))


def test_changing_layer_color_mode_updates_combo_box(qtbot):
    """Updating layer color mode changes the combo box selection"""
    layer = Labels(_LABELS)
    qtctrl = QtLabelsControls(layer)
    qtbot.addWidget(qtctrl)

    original_color_mode = layer.color_mode
    assert original_color_mode == qtctrl.colorModeComboBox.currentText()

    layer.color_mode = 'selected'
    assert layer.color_mode == qtctrl.colorModeComboBox.currentText()


def test_changing_combo_box_updates_layer_color_mode(qtbot):
    """Selecting a new color mode via combo box updates layer's color mode"""
    layer = Labels(_LABELS)
    qtctrl = QtLabelsControls(layer)
    qtbot.addWidget(qtctrl)
    color_mode_comboBox = qtctrl.colorModeComboBox

    index = color_mode_comboBox.findText('selected', Qt.MatchFixedString)
    color_mode_comboBox.setCurrentIndex(index)
    assert layer.color_mode == color_mode_comboBox.currentText()
