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


def test_qt_dask_settings_widget_value_invalid(qtbot, dask_settings_widget):
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
