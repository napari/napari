import pytest

from napari._qt.widgets.qt_size_preview import (
    QtFontSizePreview,
    QtSizeSliderPreviewWidget,
)


@pytest.fixture
def preview_widget(qtbot):
    def _preview_widget(**kwargs):
        widget = QtFontSizePreview(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _preview_widget


@pytest.fixture
def font_size_preview_widget(qtbot):
    def _font_size_preview_widget(**kwargs):
        widget = QtSizeSliderPreviewWidget(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _font_size_preview_widget


# QtFontSizePreview
# ----------------------------------------------------------------------------
def test_qt_font_size_preview_defaults(preview_widget):
    preview_widget()


def test_qt_font_size_preview_text(preview_widget):
    text = "Some text"
    widget = preview_widget(text=text)
    assert widget.text() == text

    widget = preview_widget()
    widget.setText(text)
    assert widget.text() == text


# QtSizeSliderPreviewWidget
# ----------------------------------------------------------------------------
def test_qt_size_slider_preview_widget_defaults(font_size_preview_widget):
    font_size_preview_widget()


def test_qt_size_slider_preview_widget_description(font_size_preview_widget):
    description = "Some text"
    widget = font_size_preview_widget(description=description)
    assert widget.description() == description

    widget = font_size_preview_widget()
    widget.setDescription(description)
    assert widget.description() == description


def test_qt_size_slider_preview_widget_unit(font_size_preview_widget):
    unit = "EM"
    widget = font_size_preview_widget(unit=unit)
    assert widget.unit() == unit

    widget = font_size_preview_widget()
    widget.setUnit(unit)
    assert widget.unit() == unit


def test_qt_size_slider_preview_widget_preview(font_size_preview_widget):
    preview = "Some preview"
    widget = font_size_preview_widget(preview_text=preview)
    assert widget.previewText() == preview

    widget = font_size_preview_widget()
    widget.setPreviewText(preview)
    assert widget.previewText() == preview


def test_qt_size_slider_preview_widget_minimum(font_size_preview_widget):
    minimum = 10
    widget = font_size_preview_widget(min_value=minimum)
    assert widget.minimum() == minimum
    assert widget.value() >= minimum

    widget = font_size_preview_widget()
    widget.setMinimum(5)
    assert widget.minimum() == 5
    assert widget._slider.minimum() == 5
    assert widget._slider_min_label.text() == "5"

    widget.setMinimum(20)
    assert widget.minimum() == 20
    assert widget.value() == 20
    assert widget._slider.minimum() == 20
    assert widget._slider_min_label.text() == "20"
    assert widget._lineedit.text() == "20"


def test_qt_size_slider_preview_widget_minimum_invalid(
    font_size_preview_widget,
):
    widget = font_size_preview_widget()

    with pytest.raises(ValueError):
        widget.setMinimum(60)


def test_qt_size_slider_preview_widget_maximum(font_size_preview_widget):
    maximum = 10
    widget = font_size_preview_widget(max_value=maximum)

    assert widget.maximum() == maximum
    assert widget.value() <= maximum

    widget = font_size_preview_widget()
    widget.setMaximum(20)
    assert widget.maximum() == 20
    assert widget._slider.maximum() == 20
    assert widget._slider_max_label.text() == "20"

    widget.setMaximum(5)
    assert widget.maximum() == 5
    assert widget.value() == 5
    assert widget._slider.maximum() == 5
    assert widget._lineedit.text() == "5"
    assert widget._slider_max_label.text() == "5"


def test_qt_size_slider_preview_widget_maximum_invalid(
    font_size_preview_widget,
):
    widget = font_size_preview_widget()

    with pytest.raises(ValueError):
        widget.setMaximum(-5)


def test_qt_size_slider_preview_widget_value(font_size_preview_widget):
    widget = font_size_preview_widget(value=5)
    assert widget.value() <= 5

    widget = font_size_preview_widget()
    widget.setValue(5)
    assert widget.value() == 5


def test_qt_size_slider_preview_widget_value_invalid(
    qtbot, font_size_preview_widget
):
    widget = font_size_preview_widget()
    widget.setMaximum(50)
    widget.setValue(51)
    assert widget.value() == 50
    assert widget._lineedit.text() == "50"

    widget.setMinimum(5)
    widget.setValue(1)
    assert widget.value() == 5
    assert widget._lineedit.text() == "5"


def test_qt_size_slider_preview_signal(qtbot, font_size_preview_widget):
    widget = font_size_preview_widget()

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(7)

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget.setValue(-5)
