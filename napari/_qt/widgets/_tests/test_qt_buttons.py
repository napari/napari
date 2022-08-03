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
