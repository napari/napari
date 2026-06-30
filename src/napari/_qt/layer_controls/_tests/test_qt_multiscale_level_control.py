import numpy as np
import pytest

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.layer_controls.widgets.qt_multiscale_level_control import (
    _format_level_label,
)
from napari.layers import Image, Labels


# ---------------------------------------------------------------------------
# _format_level_label
# ---------------------------------------------------------------------------
def test_format_level_label_2d():
    label = _format_level_label(0, (256, 256), 256 * 256)
    assert label == '0: 256 \u00d7 256 (65.5 KB)'


def test_format_level_label_3d():
    label = _format_level_label(2, (64, 128, 128), 64 * 128 * 128)
    assert label == '2: 64 \u00d7 128 \u00d7 128 (1.0 MB)'


def test_format_level_label_4d():
    """Full shape is always shown."""
    label = _format_level_label(1, (3, 10, 64, 64), 3 * 10 * 64 * 64)
    assert label == '1: 3 \u00d7 10 \u00d7 64 \u00d7 64 (122.9 KB)'


def test_format_level_label_1d():
    label = _format_level_label(0, (1024,), 1024)
    assert label == '0: 1024 (1.0 KB)'


# ---------------------------------------------------------------------------
# QtMultiscaleLevelControl widget tests
# ---------------------------------------------------------------------------
_MULTISCALE_DATA = [
    np.zeros((40, 20), dtype=np.uint8),
    np.zeros((20, 10), dtype=np.uint8),
    np.zeros((10, 5), dtype=np.uint8),
]


@pytest.fixture(
    params=[
        (Image, QtImageControls),
        (Labels, QtLabelsControls),
    ],
    ids=['Image', 'Labels'],
)
def multiscale_controls(qtbot, request):
    LayerCls, ControlsCls = request.param
    layer = LayerCls(_MULTISCALE_DATA, multiscale=True)
    qtctrl = ControlsCls(layer)
    qtbot.addWidget(qtctrl)
    return layer, qtctrl


def test_combobox_populated(multiscale_controls):
    """Combobox should have 'Auto' plus one entry per level."""
    _layer, qtctrl = multiscale_controls
    combo = qtctrl._multiscale_level_control.level_combobox
    assert combo.count() == 1 + len(_MULTISCALE_DATA)
    assert combo.itemText(0) == 'Auto'
    assert combo.itemData(0) is None
    for i in range(len(_MULTISCALE_DATA)):
        assert combo.itemData(i + 1) == i


def test_combobox_starts_on_auto(multiscale_controls):
    """Combobox should default to 'Auto'."""
    layer, qtctrl = multiscale_controls
    combo = qtctrl._multiscale_level_control.level_combobox
    assert combo.currentIndex() == 0
    assert layer.locked_data_level is None


def test_selecting_level_sets_locked_data_level(multiscale_controls):
    """Choosing a level in the combobox should set locked_data_level."""
    layer, qtctrl = multiscale_controls
    combo = qtctrl._multiscale_level_control.level_combobox
    combo.setCurrentIndex(2)  # level 1
    assert layer.locked_data_level == 1


def test_selecting_auto_clears_locked_data_level(multiscale_controls):
    """Switching back to 'Auto' should set locked_data_level to None."""
    layer, qtctrl = multiscale_controls
    combo = qtctrl._multiscale_level_control.level_combobox
    combo.setCurrentIndex(1)
    assert layer.locked_data_level == 0

    combo.setCurrentIndex(0)
    assert layer.locked_data_level is None


def test_programmatic_change_updates_combobox(multiscale_controls):
    """Setting locked_data_level on the layer should update the combobox."""
    layer, qtctrl = multiscale_controls
    combo = qtctrl._multiscale_level_control.level_combobox
    assert combo.currentIndex() == 0

    layer.locked_data_level = 2
    assert combo.currentIndex() == 3  # +1 for "Auto"

    layer.locked_data_level = None
    assert combo.currentIndex() == 0


def test_combobox_updates_on_data_change(multiscale_controls):
    """Replacing layer.data with fewer levels should update the combobox."""
    layer, qtctrl = multiscale_controls
    combo = qtctrl._multiscale_level_control.level_combobox
    # Initially 3 levels + Auto
    assert combo.count() == 1 + len(_MULTISCALE_DATA)

    # Replace with a 2-level multiscale
    new_data = [
        np.zeros((20, 10), dtype=np.uint8),
        np.zeros((10, 5), dtype=np.uint8),
    ]
    layer.data = new_data

    # Combobox should now have Auto + 2 levels
    assert combo.count() == 1 + len(new_data)
    for i in range(len(new_data)):
        assert combo.itemData(i + 1) == i


def test_not_multiscale_is_hidden(qtbot):
    """Single-scale layer should hide the multiscale control."""
    layer = Image(np.zeros((40, 20), dtype=np.uint8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    ctrl = qtctrl._multiscale_level_control
    assert ctrl.level_combobox.isHidden()
    assert ctrl.level_label.isHidden()


def test_multiscale_labels_show_full_shape(qtbot):
    """Labels should show the full shape of each level."""
    data = [
        np.zeros((3, 10, 64, 64), dtype=np.uint8),
        np.zeros((3, 10, 32, 32), dtype=np.uint8),
    ]
    layer = Image(data, multiscale=True)
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    combo = qtctrl._multiscale_level_control.level_combobox

    # Level 0: (3, 10, 64, 64) = 122880 bytes
    label_0 = combo.itemText(1)
    assert '3 \u00d7 10 \u00d7 64 \u00d7 64' in label_0
    assert '122.9 KB' in label_0 or '120.0 KB' in label_0

    # Level 1: (3, 10, 32, 32) = 30720 bytes
    label_1 = combo.itemText(2)
    assert '3 \u00d7 10 \u00d7 32 \u00d7 32' in label_1
    assert '30.7 KB' in label_1 or '30.0 KB' in label_1
