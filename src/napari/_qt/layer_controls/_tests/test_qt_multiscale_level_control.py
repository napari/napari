from unittest.mock import patch

import numpy as np
import pytest
from qtpy.QtGui import QStandardItemModel

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.layer_controls.widgets.qt_multiscale_level_control import (
    _format_level_label,
    _human_readable_size,
)
from napari.layers import Image, Labels


# ---------------------------------------------------------------------------
# _human_readable_size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ('size_bytes', 'expected'),
    [
        (0, '0.0 B'),
        (1, '1.0 B'),
        (999, '999.0 B'),
        (1000, '1.0 KB'),
        (1500, '1.5 KB'),
        (1_000_000, '1.0 MB'),
        (1_500_000, '1.5 MB'),
        (1_000_000_000, '1.0 GB'),
        (1_000_000_000_000, '1.0 TB'),
        (1_000_000_000_000_000, '1.0 PB'),
        (2_500_000_000_000_000, '2.5 PB'),
    ],
)
def test_human_readable_size(size_bytes, expected):
    assert _human_readable_size(size_bytes) == expected


# ---------------------------------------------------------------------------
# _format_level_label
# ---------------------------------------------------------------------------
def test_format_level_label_2d():
    label = _format_level_label(0, (256, 256), 256 * 256)
    assert label == '0: 256 \u00d7 256 (65.5 KB)'


def test_format_level_label_3d():
    label = _format_level_label(2, (64, 128, 128), 64 * 128 * 128)
    assert label == '2: 64 \u00d7 128 \u00d7 128 (1.0 MB)'


def test_format_level_label_no_displayed_axes_shows_full_shape():
    """Without displayed_axes the full shape is shown."""
    label = _format_level_label(1, (3, 10, 64, 64), 3 * 10 * 64 * 64)
    assert label == '1: 3 \u00d7 10 \u00d7 64 \u00d7 64 (122.9 KB)'


def test_format_level_label_displayed_axes():
    """Only the displayed dimensions should appear in the label."""
    label = _format_level_label(
        0, (5, 10, 64, 64), 5 * 10 * 64 * 64, displayed_axes=(2, 3)
    )
    assert label == '0: 64 \u00d7 64 (204.8 KB)'


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


def test_not_multiscale_has_only_auto_and_is_hidden(qtbot):
    """Single-scale layer should have only 'Auto' and hide the control."""
    layer = Image(np.zeros((40, 20), dtype=np.uint8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    ctrl = qtctrl._multiscale_level_control
    assert ctrl.level_combobox.count() == 1
    assert ctrl.level_combobox.itemText(0) == 'Auto'
    assert ctrl.level_combobox.isHidden()
    assert ctrl.level_label.isHidden()


# ---------------------------------------------------------------------------
# GL texture limit tests
# ---------------------------------------------------------------------------
_3D_MULTISCALE_DATA = [
    np.zeros((64, 64, 64), dtype=np.uint8),
    np.zeros((32, 32, 32), dtype=np.uint8),
    np.zeros((16, 16, 16), dtype=np.uint8),
]


def test_levels_exceeding_3d_texture_limit_are_disabled(make_napari_viewer):
    """In 3D, levels whose shape exceeds GL_MAX_3D_TEXTURE_SIZE are disabled."""
    viewer = make_napari_viewer()
    layer = viewer.add_image(_3D_MULTISCALE_DATA, multiscale=True)

    with patch(
        'napari._qt.layer_controls.widgets.qt_multiscale_level_control.get_max_texture_sizes',
        return_value=(16384, 32),
    ):
        viewer.dims.ndisplay = 3

    ctrl = viewer.window._qt_viewer.controls.widgets[layer]
    combo = ctrl._multiscale_level_control.level_combobox
    model = combo.model()
    assert isinstance(model, QStandardItemModel)

    # "Auto" (index 0) should be enabled
    assert model.item(0).isEnabled()
    # Level 0 (64^3) exceeds limit of 32 — disabled
    assert not model.item(1).isEnabled()
    # Level 1 (32^3) fits — enabled
    assert model.item(2).isEnabled()
    # Level 2 (16^3) fits — enabled
    assert model.item(3).isEnabled()


def test_levels_all_enabled_in_2d(make_napari_viewer):
    """In 2D, no levels should be disabled regardless of 3D texture limit."""
    viewer = make_napari_viewer()
    layer = viewer.add_image(_3D_MULTISCALE_DATA, multiscale=True)

    with patch(
        'napari._qt.layer_controls.widgets.qt_multiscale_level_control.get_max_texture_sizes',
        return_value=(16384, 32),
    ):
        # Force a rebuild while in 2D
        ctrl = viewer.window._qt_viewer.controls.widgets[layer]
        ctrl._multiscale_level_control._rebuild_items()

    combo = ctrl._multiscale_level_control.level_combobox
    model = combo.model()
    assert isinstance(model, QStandardItemModel)

    for i in range(combo.count()):
        assert model.item(i).isEnabled()
