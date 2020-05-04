from napari._qt.qt_plugin_report import QtPluginErrReporter
from napari_plugin_engine import PluginError
import pytest


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_error_reporter(qtbot, test_plugin_manager):
    """test that QtPluginErrReporter shows any instantiated PluginErrors."""
    error_message = 'my special error'
    _ = PluginError(error_message, plugin_name='plugin_name')
    report = QtPluginErrReporter(plugin_manager=test_plugin_manager)
    qtbot.addWidget(report)

    # the null option plus the one we created
    assert report.plugin_combo.count() >= 2

    # the message should appear somewhere in the text area
    report.set_plugin('plugin_name')
    assert error_message in report.text_area.toPlainText()

    # plugins without errors raise an error
    with pytest.raises(ValueError):
        report.set_plugin('non_existent')
