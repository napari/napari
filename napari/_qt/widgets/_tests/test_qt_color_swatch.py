import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from napari._qt.widgets.qt_color_swatch import (
    TRANSPARENT,
    QColorPopup,
    QColorSwatch,
    QColorSwatchEdit,
)


@pytest.mark.parametrize('color', [None, [1, 1, 1, 1]])
@pytest.mark.parametrize('tooltip', [None, 'This is a test'])
def test_succesfull_create_qcolorswatchedit(qtbot, color, tooltip):
    widget = QColorSwatchEdit(initial_color=color, tooltip=tooltip)
    qtbot.add_widget(widget)

    test_color = color or TRANSPARENT
    test_tooltip = tooltip or 'click to set color'

    # check widget creation and base values
    assert widget.color_swatch.toolTip() == test_tooltip
    np.testing.assert_array_equal(widget.color, test_color)

    # check widget popup
    qtbot.mouseRelease(widget.color_swatch, Qt.MouseButton.LeftButton)
    color_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QColorPopup):
            color_popup = widget
    assert color_popup


@pytest.mark.parametrize('color', [None, [1, 1, 1, 1]])
@pytest.mark.parametrize('tooltip', [None, 'This is a test'])
def test_succesfull_create_qcolorswatch(qtbot, color, tooltip):
    widget = QColorSwatch(initial_color=color, tooltip=tooltip)
    qtbot.add_widget(widget)

    test_color = color or TRANSPARENT
    test_tooltip = tooltip or 'click to set color'

    # check widget creation and base values
    assert widget.toolTip() == test_tooltip
    np.testing.assert_array_equal(widget.color, test_color)

    # check widget popup
    qtbot.mouseRelease(widget, Qt.MouseButton.LeftButton)
    color_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QColorPopup):
            color_popup = widget
    assert color_popup
