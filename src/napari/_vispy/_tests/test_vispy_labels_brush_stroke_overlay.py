import numpy as np
import pytest

from napari._vispy.overlays.labels_brush_stroke import (
    RADIUS_FACTOR,
    VispyLabelsBrushStrokeOverlay,
)
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.layers.labels._labels_key_bindings import reset_polygon
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
)


def _make_overlay():
    viewer = ViewerModel()
    data = np.zeros((40, 40), dtype=np.int32)
    layer = viewer.add_labels(data)
    overlay = layer._overlays['brush_stroke']
    vispy = VispyLabelsBrushStrokeOverlay(
        layer=layer,
        font_info=FontInfo(),
        viewer=viewer,
        overlay=overlay,
    )
    layer.mode = 'paint'  # enables the overlay
    layer.selected_label = 1
    layer.brush_size = 2
    return layer, overlay, vispy, data


def _press(MouseEvent, layer, position):
    event = MouseEvent(
        type='mouse_press', button=2, position=position, dims_displayed=(0, 1)
    )
    event.modifiers = ()
    mouse_press_callbacks(layer, event)


def _move(MouseEvent, layer, position):
    event = MouseEvent(
        type='mouse_move', position=position, dims_displayed=(0, 1)
    )
    event.modifiers = ()
    mouse_move_callbacks(layer, event)


@pytest.mark.usefixtures('qapp')
def test_brush_stroke_circle_appears_on_right_click(MouseEvent):
    layer, overlay, vispy, data = _make_overlay()

    _press(MouseEvent, layer, (15, 15))

    assert overlay.active is True
    assert vispy._circle.visible is True
    assert overlay.radius == layer.brush_size * RADIUS_FACTOR
    # first pixel painted
    assert data[15, 15] == 1


@pytest.mark.usefixtures('qapp')
def test_live_paint_on_move_is_staged_not_committed(MouseEvent):
    layer, overlay, vispy, data = _make_overlay()

    _press(MouseEvent, layer, (15, 15))
    # moves staying within the radius do not complete the stroke
    for position in [(15, 16), (16, 16), (16, 15)]:
        _move(MouseEvent, layer, position)

    assert overlay.active is True
    assert layer._block_history is True
    assert len(layer._staged_history) > 0
    assert len(layer._undo_history) == 0
    assert data[16, 16] == 1


@pytest.mark.usefixtures('qapp')
def test_escape_aborts_and_restores(MouseEvent):
    layer, overlay, vispy, data = _make_overlay()

    _press(MouseEvent, layer, (15, 15))
    for position in [(15, 16), (16, 16)]:
        _move(MouseEvent, layer, position)

    # Escape -> abort
    reset_polygon(layer)

    assert overlay.active is False
    assert vispy._circle.visible is False
    assert np.array_equiv(data, 0)
    assert layer._staged_history == []
    assert layer._block_history is False
    assert len(layer._undo_history) == 0


@pytest.mark.usefixtures('qapp')
def test_leave_and_return_completes_and_fills(MouseEvent):
    layer, overlay, vispy, data = _make_overlay()

    _press(MouseEvent, layer, (15, 15))
    # trace a square loop: leave the radius, then return to the start
    loop = [
        (8, 15),  # leaves the radius -> latches _has_left
        (8, 22),
        (22, 22),
        (22, 8),
        (8, 8),
        (8, 15),
        (13, 16),  # back inside the radius (but not the start) -> completes
    ]
    for position in loop:
        _move(MouseEvent, layer, position)

    assert overlay.active is False
    assert vispy._circle.visible is False
    # interior pixel (not on the brush path) is filled by paint_polygon
    assert data[15, 18] == 1
    # whole stroke committed as a single undo item
    assert len(layer._undo_history) == 1

    layer.undo()
    assert np.array_equiv(data, 0)
