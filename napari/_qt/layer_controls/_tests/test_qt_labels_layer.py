import numpy as np

from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels

np.random.seed(0)
_LABELS = np.random.randint(5, size=(10, 15))
_COLOR = {1: 'white', 2: 'blue', 3: 'green', 4: 'red', 5: 'yellow'}


def test_changing_layer_color_mode_updates_combo_box(qtbot):
    """Updating layer color mode changes the combo box selection"""
    layer = Labels(_LABELS, color=_COLOR)
    qtctrl = QtLabelsControls(layer)
    qtbot.addWidget(qtctrl)

    original_color_mode = layer.color_mode
    assert original_color_mode == qtctrl.colorModeComboBox.currentText()

    layer.color_mode = 'auto'
    assert layer.color_mode == qtctrl.colorModeComboBox.currentText()
