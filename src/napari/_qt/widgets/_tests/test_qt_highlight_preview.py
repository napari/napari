import pytest

from napari._qt.widgets.qt_highlight_preview import (
    QtHighlightSizePreviewWidget,
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
def highlight_size_preview_widget(qtbot):
    def _highlight_size_preview_widget(**kwargs):
        widget = QtHighlightSizePreviewWidget(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _highlight_size_preview_widget


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


# QtHighlightSizePreviewWidget
# ----------------------------------------------------------------------------


def test_qt_highlight_size_preview_widget_defaults(
    highlight_size_preview_widget,
):
    highlight_size_preview_widget()


def test_qt_highlight_size_preview_widget_description(
    highlight_size_preview_widget,
):
    description = "Some text"
    widget = highlight_size_preview_widget(description=description)
    assert widget.description() == description

    widget = highlight_size_preview_widget()
    widget.setDescription(description)
    assert widget.description() == description


def test_qt_highlight_size_preview_widget_unit(highlight_size_preview_widget):
    unit = "CM"
    widget = highlight_size_preview_widget(unit=unit)
    assert widget.unit() == unit

    widget = highlight_size_preview_widget()
    widget.setUnit(unit)
    assert widget.unit() == unit


def test_qt_highlight_size_preview_widget_minimum(
    highlight_size_preview_widget,
):
    minimum = 5
    widget = highlight_size_preview_widget(min_value=minimum)
    assert widget.minimum() == minimum
    assert widget.value() >= minimum

    widget = highlight_size_preview_widget()
    widget.setMinimum(3)
    assert widget.minimum() == 3
    assert widget.value() == 3
    assert widget._slider.minimum() == 3
    assert widget._slider_min_label.text() == "3"
    assert widget._triangle.minimum() == 3
    assert widget._lineedit.text() == "3"


def test_qt_highlight_size_preview_widget_minimum_invalid(
    highlight_size_preview_widget,
):
    widget = highlight_size_preview_widget()

    with pytest.raises(ValueError):
        widget.setMinimum(60)


def test_qt_highlight_size_preview_widget_maximum(
    highlight_size_preview_widget,
):
    maximum = 10
    widget = highlight_size_preview_widget(max_value=maximum)

    assert widget.maximum() == maximum
    assert widget.value() <= maximum

    widget = highlight_size_preview_widget()
    widget.setMaximum(20)
    assert widget.maximum() == 20
    assert widget._slider.maximum() == 20
    assert widget._triangle.maximum() == 20
    assert widget._slider_max_label.text() == "20"

    widget.setMaximum(5)
    assert widget.maximum() == 5
    # assert widget.value() == 5
    # assert widget._slider.maximum() == 5
    # assert widget._triangle.maximum() == 20
    # assert widget._lineedit.text() == "5"
    # assert widget._slider_max_label.text() == "5"


def test_qt_highlight_size_preview_widget_maximum_invalid(
    highlight_size_preview_widget,
):
    widget = highlight_size_preview_widget()

    with pytest.raises(ValueError):
        widget.setMaximum(-5)


def test_qt_highlight_size_preview_widget_value(highlight_size_preview_widget):
    widget = highlight_size_preview_widget(value=5)
    assert widget.value() <= 5

    widget = highlight_size_preview_widget()
    widget.setValue(5)
    assert widget.value() == 5


def test_qt_highlight_size_preview_widget_value_invalid(
    qtbot, highlight_size_preview_widget
):
    widget = highlight_size_preview_widget()
    widget.setMaximum(50)
    widget.setValue(51)
    assert widget.value() == 50
    assert widget._lineedit.text() == "50"

    widget.setMinimum(5)
    widget.setValue(1)
    assert widget.value() == 5
    assert widget._lineedit.text() == "5"


def test_qt_highlight_size_preview_widget_signal(
    qtbot, highlight_size_preview_widget
):
    widget = highlight_size_preview_widget()

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(7)

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(-5)
