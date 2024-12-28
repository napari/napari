import numpy as np
import pytest

from napari._qt.widgets.qt_highlight_preview import (
    QtHighlightPreviewWidget,
    QtStar,
    QtTriangle,
)


@pytest.fixture
def star_widget(qtbot):
    def _star_widget(**kwargs):
        widget = QtStar(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _star_widget


@pytest.fixture
def triangle_widget(qtbot):
    def _triangle_widget(**kwargs):
        widget = QtTriangle(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _triangle_widget


@pytest.fixture
def highlight_preview_widget(qtbot):
    def _highlight_preview_widget(**kwargs):
        widget = QtHighlightPreviewWidget(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _highlight_preview_widget


# QtStar
# ----------------------------------------------------------------------------


def test_qt_star_defaults(star_widget):
    star_widget()


def test_qt_star_value(star_widget):
    widget = star_widget(value=5)
    assert widget.value() <= 5

    widget = star_widget()
    widget.setValue(5)
    assert widget.value() == 5


# QtTriangle
# ----------------------------------------------------------------------------


def test_qt_triangle_defaults(triangle_widget):
    triangle_widget()


def test_qt_triangle_value(triangle_widget):
    widget = triangle_widget(value=5)
    assert widget.value() <= 5

    widget = triangle_widget()
    widget.setValue(5)
    assert widget.value() == 5


def test_qt_triangle_minimum(triangle_widget):
    minimum = 1
    widget = triangle_widget(min_value=minimum)
    assert widget.minimum() == minimum
    assert widget.value() >= minimum

    widget = triangle_widget()
    widget.setMinimum(2)
    assert widget.minimum() == 2
    assert widget.value() == 2


def test_qt_triangle_maximum(triangle_widget):
    maximum = 10
    widget = triangle_widget(max_value=maximum)
    assert widget.maximum() == maximum
    assert widget.value() <= maximum

    widget = triangle_widget()
    widget.setMaximum(20)
    assert widget.maximum() == 20


def test_qt_triangle_signal(qtbot, triangle_widget):
    widget = triangle_widget()

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(7)

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(-5)


# QtHighlightPreviewWidget
# ----------------------------------------------------------------------------


def test_qt_highlight_preview_widget_defaults(
    highlight_preview_widget,
):
    highlight_preview_widget()


def test_qt_highlight_preview_widget_description(
    highlight_preview_widget,
):
    description = 'Some text'
    widget = highlight_preview_widget(description=description)
    assert widget.description() == description

    widget = highlight_preview_widget()
    widget.setDescription(description)
    assert widget.description() == description


def test_qt_highlight_preview_widget_unit(highlight_preview_widget):
    unit = 'CM'
    widget = highlight_preview_widget(unit=unit)
    assert widget.unit() == unit

    widget = highlight_preview_widget()
    widget.setUnit(unit)
    assert widget.unit() == unit


def test_qt_highlight_preview_widget_minimum(
    highlight_preview_widget,
):
    minimum = 5
    widget = highlight_preview_widget(min_value=minimum)
    assert widget.minimum() == minimum
    assert widget._thickness_value >= minimum
    assert widget.value()['highlight_thickness'] >= minimum

    widget = highlight_preview_widget()
    widget.setMinimum(3)
    assert widget.minimum() == 3
    assert widget._thickness_value == 3
    assert widget.value()['highlight_thickness'] == 3
    assert widget._slider.minimum() == 3
    assert widget._slider_min_label.text() == '3'
    assert widget._triangle.minimum() == 3
    assert widget._lineedit.text() == '3'


def test_qt_highlight_preview_widget_minimum_invalid(
    highlight_preview_widget,
):
    widget = highlight_preview_widget()

    with pytest.raises(ValueError, match='must be smaller than'):
        widget.setMinimum(60)


def test_qt_highlight_preview_widget_maximum(
    highlight_preview_widget,
):
    maximum = 10
    widget = highlight_preview_widget(max_value=maximum)

    assert widget.maximum() == maximum
    assert widget._thickness_value <= maximum
    assert widget.value()['highlight_thickness'] <= maximum

    widget = highlight_preview_widget(
        value={
            'highlight_thickness': 6,
            'highlight_color': [0.0, 0.6, 1.0, 1.0],
        }
    )
    widget.setMaximum(20)
    assert widget.maximum() == 20
    assert widget._slider.maximum() == 20
    assert widget._triangle.maximum() == 20
    assert widget._slider_max_label.text() == '20'

    assert widget._thickness_value == 6
    assert widget.value()['highlight_thickness'] == 6
    widget.setMaximum(5)
    assert widget.maximum() == 5
    assert widget._thickness_value == 5
    assert widget.value()['highlight_thickness'] == 5
    assert widget._slider.maximum() == 5
    assert widget._triangle.maximum() == 5
    assert widget._lineedit.text() == '5'
    assert widget._slider_max_label.text() == '5'


def test_qt_highlight_preview_widget_maximum_invalid(
    highlight_preview_widget,
):
    widget = highlight_preview_widget()

    with pytest.raises(ValueError, match='must be larger than'):
        widget.setMaximum(-5)


def test_qt_highlight_preview_widget_value(highlight_preview_widget):
    widget = highlight_preview_widget(
        value={
            'highlight_thickness': 5,
            'highlight_color': [0.0, 0.6, 1.0, 1.0],
        }
    )
    assert widget._thickness_value <= 5
    assert widget.value()['highlight_thickness'] <= 5
    assert widget._color_value == [0.0, 0.6, 1.0, 1.0]
    assert widget.value()['highlight_color'] == [0.0, 0.6, 1.0, 1.0]

    widget = highlight_preview_widget()
    widget.setValue(
        {'highlight_thickness': 5, 'highlight_color': [0.6, 0.6, 1.0, 1.0]}
    )
    assert widget._thickness_value == 5
    assert widget.value()['highlight_thickness'] == 5
    assert np.array_equal(
        np.array(widget._color_value, dtype=np.float32),
        np.array([0.6, 0.6, 1.0, 1.0], dtype=np.float32),
    )
    assert np.array_equal(
        np.array(widget.value()['highlight_color'], dtype=np.float32),
        np.array([0.6, 0.6, 1.0, 1.0], dtype=np.float32),
    )


def test_qt_highlight_preview_widget_value_invalid(
    qtbot, highlight_preview_widget
):
    widget = highlight_preview_widget()
    widget.setMaximum(50)
    widget.setValue(
        {'highlight_thickness': 51, 'highlight_color': [0.0, 0.6, 1.0, 1.0]}
    )
    assert widget.value()['highlight_thickness'] == 50
    assert widget._lineedit.text() == '50'

    widget.setMinimum(5)
    widget.setValue(
        {'highlight_thickness': 1, 'highlight_color': [0.0, 0.6, 1.0, 1.0]}
    )
    assert widget.value()['highlight_thickness'] == 5
    assert widget._lineedit.text() == '5'


def test_qt_highlight_preview_widget_signal(qtbot, highlight_preview_widget):
    widget = highlight_preview_widget()

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(
            {'highlight_thickness': 7, 'highlight_color': [0.0, 0.6, 1.0, 1.0]}
        )

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(
            {
                'highlight_thickness': -5,
                'highlight_color': [0.0, 0.6, 1.0, 1.0],
            }
        )
