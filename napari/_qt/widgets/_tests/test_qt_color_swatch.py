import numpy
import pytest

from napari._qt.widgets.qt_color_swatch import (
    TRANSPARENT,
    QColorSwatch,
    QColorSwatchEdit,
)


@pytest.mark.parametrize('color', [None, [1, 1, 1, 1]])
@pytest.mark.parametrize('tooltip', [None, 'This is a test'])
def test_succesfull_create_qcolorswatchedit(qtbot, color, tooltip):
    widget = QColorSwatchEdit(initial_color=color, tooltip=tooltip)
    qtbot.add_widget(widget)
    if color is None:
        test_color = TRANSPARENT
    else:
        test_color = color

    if tooltip is None:
        test_tooltip = 'click to set color'
    else:
        test_tooltip = tooltip

    assert widget.color_swatch.toolTip() == test_tooltip
    numpy.testing.assert_array_equal(widget.color, test_color)


@pytest.mark.parametrize('color', [None, [1, 1, 1, 1]])
@pytest.mark.parametrize('tooltip', [None, 'This is a test'])
def test_succesfull_create_qcolorswatch(qtbot, color, tooltip):
    widget = QColorSwatch(initial_color=color, tooltip=tooltip)
    qtbot.add_widget(widget)
    if color is None:
        test_color = TRANSPARENT
    else:
        test_color = color

    if tooltip is None:
        test_tooltip = 'click to set color'
    else:
        test_tooltip = tooltip

    assert widget.toolTip() == test_tooltip
    numpy.testing.assert_array_equal(widget.color, test_color)
