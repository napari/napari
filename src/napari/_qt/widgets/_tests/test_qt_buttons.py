from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from napari.layers import Points
from napari.layers.points._points_constants import Mode


def test_radio_button(qtbot):
    """Make sure the QtModeRadioButton works to change layer modes"""
    layer = Points()
    assert layer.mode != Mode.ADD

    btn = QtModeRadioButton(layer, 'test_button', Mode.ADD, tooltip='tooltip')
    assert btn.property('mode') == 'test_button'
    assert btn.toolTip() == 'tooltip'

    btn.click()
    qtbot.waitUntil(lambda: layer.mode == Mode.ADD, timeout=500)


def test_push_button(qtbot):
    """Make sure the QtModePushButton works with callbacks"""
    layer = Points()
    layer.test_prop = False

    def set_test_prop():
        layer.test_prop = True

    btn = QtModePushButton(
        layer, 'test_button', slot=set_test_prop, tooltip='tooltip'
    )
    assert btn.property('mode') == 'test_button'
    assert btn.toolTip() == 'tooltip'

    btn.click()
    qtbot.waitUntil(lambda: layer.test_prop, timeout=500)


def test_push_button_can_toggle(qtbot):
    """Make sure the QtModePushButton supports toggle callbacks."""
    layer = Points()
    layer.test_states = []

    def record_state(state):
        layer.test_states.append(state)

    btn = QtModePushButton(
        layer,
        'test_button',
        slot=record_state,
        tooltip='tooltip',
        checkable=True,
    )
    assert btn.property('mode') == 'test_button'
    assert btn.toolTip() == 'tooltip'
    assert btn.isCheckable()

    btn.click()
    qtbot.waitUntil(lambda: layer.test_states == [True], timeout=500)

    btn.click()
    qtbot.waitUntil(lambda: layer.test_states == [True, False], timeout=500)


def test_layers_button_works(make_napari_viewer):
    v = make_napari_viewer()
    layer = v.add_layer(Points())
    assert layer.mode != 'add'
    controls = v.window._qt_viewer.controls.widgets[layer]
    controls.addition_button.click()
    assert layer.mode == 'add'
