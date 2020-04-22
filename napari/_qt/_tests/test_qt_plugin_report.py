from napari._qt.qt_plugin_report import QtPluginErrReporter
from naplugi import PluginError
import pytest


def test_error_reporter(qtbot):
    """test that QtPluginErrReporter shows any instantiated PluginErrors."""
    from napari.plugins import plugin_manager

    error_message = 'my special error'
    _ = PluginError(
        error_message, plugin_name='plugin_name', manager=plugin_manager
    )
    report = QtPluginErrReporter()
    qtbot.addWidget(report)

    # the null option plus the one we created
    assert report.plugin_combo.count() >= 2

    # the message should appear somewhere in the text area
    report.set_plugin('plugin_name')
    assert error_message in report.text_area.toPlainText()

    # plugins without errors raise an error
    with pytest.raises(ValueError):
        report.set_plugin('non_existent')
