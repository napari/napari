from unittest.mock import patch

import pytest
from qtpy.QtCore import Qt

from napari._qt.dialogs.qt_updates import UpdateOptionsDialog
from napari import settings


class DummySettings:
    class DummyUpdates:
        update_version_skip = []
    updates = DummyUpdates


def dummy_get_settings():
    DummySettings.updates.update_version_skip = []
    return DummySettings


class TestUpdateOptionsDialog:

    def test_default_args(self, qtbot):
        dlg = UpdateOptionsDialog()
        qtbot.addWidget(dlg)
        dlg.show()

    def test_version_value(self, qtbot):
        version = '1.0.0'
        dlg = UpdateOptionsDialog(version=version)
        dlg._text.text == version
        qtbot.addWidget(dlg)
        dlg.show()

    def test_button_update(self, qtbot):
        dlg = UpdateOptionsDialog()
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.mouseClick(dlg._button_update, Qt.LeftButton)
        assert dlg.is_update()

    def test_button_update_on_quitdefault_args(self, qtbot):
        dlg = UpdateOptionsDialog()
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.mouseClick(dlg._button_update_on_quit, Qt.LeftButton)
        assert dlg.is_update_on_quit()

    def test_button_dismiss(self, qtbot):
        dlg = UpdateOptionsDialog()
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.mouseClick(dlg._button_dismiss, Qt.LeftButton)
        assert not dlg.isVisible()

    def test_button_skip_no_duplicates(self, qtbot):
        with patch.object(settings, 'get_settings', new=dummy_get_settings):
            version = '0.1.1'
            _settings = settings.get_settings()
            _settings.updates.update_version_skip = [version]

            dlg = UpdateOptionsDialog(version=version)
            qtbot.addWidget(dlg)
            dlg.show()
            qtbot.mouseClick(dlg._button_skip, Qt.LeftButton)
            assert _settings.updates.update_version_skip == [version]

    @pytest.mark.parametrize('version,expected', [('0.1.1', ['0.1.1']), ('', [])])
    def test_button_skip(self, qtbot, version, expected):
        with patch.object(settings, 'get_settings', new=dummy_get_settings):
            _settings = settings.get_settings()
            dlg = UpdateOptionsDialog(version=version)
            qtbot.addWidget(dlg)
            dlg.show()
            qtbot.mouseClick(dlg._button_skip, Qt.LeftButton)
            print(version, expected)
            assert _settings.updates.update_version_skip == expected
