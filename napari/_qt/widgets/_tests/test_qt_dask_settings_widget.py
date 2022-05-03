import pytest

from napari._qt.widgets.qt_dask_settings import QtDaskSettingsWidget


@pytest.fixture
def dask_settings_widget(qtbot):
    def _dask_settings_widget(**kwargs):
        widget = QtDaskSettingsWidget(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _dask_settings_widget


def test_qt_dask_settings_widget_defaults(
    dask_settings_widget,
):
    dask_settings_widget()


def test_qt_dask_settings_widget_cache(dask_settings_widget):
    widget = dask_settings_widget(cache=5)
    assert widget.cacheValue() == 5

    widget = dask_settings_widget()
    widget._update_cache(5)
    assert widget.cacheValue() == 5


def test_qt_dask_settings_widget_cache_value_invalid(
    qtbot, dask_settings_widget
):
    widget = dask_settings_widget()
    widget.setMaximum(500)
    widget._update_cache(510)
    assert widget.cacheValue() == 500
    assert widget._unit_label.text() == '/500 mb'

    widget.setMinimum(5)
    widget._update_cache(1)
    assert widget.cacheValue() == 5

    with pytest.raises(ValueError):
        widget.setMaximum(-5)


def test_qt_dask_settings_set_unit(dask_settings_widget):
    widget = dask_settings_widget()
    widget.set_unit('kb')

    assert widget._unit_label.text() == '/10 kb'


def test_qt_dask_settings_widget_signal(qtbot, dask_settings_widget):
    widget = dask_settings_widget()

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget._update_cache(6)

    with qtbot.waitSignal(widget.valueChanged, timeout=500):
        widget._enabled_checkbox.setChecked(False)


def test_qt_dask_settings_widget_description(dask_settings_widget):
    description = "Some text"
    widget = dask_settings_widget(description=description)
    assert widget._description == description

    widget = dask_settings_widget()
    widget.setDescription(description)
    assert widget._description == description


def test_qt_dask_settings_widget_value(dask_settings_widget):
    value = {'enabled': False, 'cache': 5}
    widget = dask_settings_widget(value=value)
    assert widget.value() == value

    widget = dask_settings_widget()
    widget.setValue(value)
    assert widget.value() == value


def test_qt_dask_settings_widget_minimum(dask_settings_widget):
    minimum = 2
    widget = dask_settings_widget(min_value=minimum)
    assert widget.minimum() == minimum
    assert widget.cacheValue() >= minimum

    widget = dask_settings_widget()
    widget.setMinimum(2)
    assert widget.minimum() == 2
    assert widget.cacheValue() == 2


def test_qt_dask_settings_widget_maximum(dask_settings_widget):
    maximum = 15
    widget = dask_settings_widget(max_value=maximum)
    assert widget.maximum() == maximum
    assert widget.cacheValue() <= maximum

    widget = dask_settings_widget()
    widget.setMaximum(20)
    assert widget.maximum() == 20


def test_qt_dask_settings_widget_set_increment(dask_settings_widget):
    increment = 2

    widget = dask_settings_widget(inc=increment)
    assert widget.increment() == increment
