import webbrowser

import pytest
from napari_plugin_engine import PluginError
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from napari._qt.dialogs import qt_plugin_report


# qtbot fixture comes from pytest-qt
# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
# monkeypatch fixture is from pytest
def test_error_reporter(qtbot, monkeypatch):
    """test that QtPluginErrReporter shows any instantiated PluginErrors."""

    monkeypatch.setattr(
        qt_plugin_report,
        'standard_metadata',
        lambda x: {'url': 'https://github.com/example/example'},
    )

    error_message = 'my special error'
    _ = PluginError(error_message, plugin_name='test_plugin', plugin="mock")
    report_widget = qt_plugin_report.QtPluginErrReporter()
    qtbot.addWidget(report_widget)

    # the null option plus the one we created
    assert report_widget.plugin_combo.count() >= 2

    # the message should appear somewhere in the text area
    report_widget.set_plugin('test_plugin')
    assert error_message in report_widget.text_area.toPlainText()

    # mock_webbrowser_open
    def mock_webbrowser_open(url, new=0):
        assert new == 2
        assert "Errors for plugin 'test_plugin'" in url
        assert "Traceback from napari" in url

    monkeypatch.setattr(webbrowser, 'open', mock_webbrowser_open)

    qtbot.mouseClick(report_widget.github_button, Qt.LeftButton)

    # make sure we can copy traceback to clipboard
    report_widget.copyToClipboard()
    clipboard_text = QGuiApplication.clipboard().text()
    assert "Errors for plugin 'test_plugin'" in clipboard_text

    # plugins without errors raise an error
    with pytest.raises(ValueError):
        report_widget.set_plugin('non_existent')

    report_widget.set_plugin(None)
    assert not report_widget.text_area.toPlainText()


def test_dialog_create(qtbot):
    dialog = qt_plugin_report.QtPluginErrReporter()
    qtbot.addWidget(dialog)
