from qtpy.QtWidgets import QDialog

from napari._qt.dialogs.confirm_close_dialog import ConfirmCloseDialog
from napari.settings import get_settings


def test_create_application_close(qtbot):
    dialog = ConfirmCloseDialog(None, close_app=True)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == 'Close Application?'
    assert get_settings().application.confirm_close_window
    assert dialog.close_btn.shortcut().toString() == 'Ctrl+Q'
    dialog.close_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert get_settings().application.confirm_close_window


def test_remove_confirmation(qtbot):
    dialog = ConfirmCloseDialog(None, close_app=True)
    dialog.do_not_ask.setChecked(True)
    assert get_settings().application.confirm_close_window
    dialog.close_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not get_settings().application.confirm_close_window


def test_remove_confirmation_reject(qtbot):
    dialog = ConfirmCloseDialog(None, close_app=True)
    dialog.do_not_ask.setChecked(True)
    assert get_settings().application.confirm_close_window
    dialog.cancel_btn.click()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert get_settings().application.confirm_close_window


def test_create_window_close(qtbot):
    dialog = ConfirmCloseDialog(None, close_app=False)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == 'Close Window?'
    assert get_settings().application.confirm_close_window
    assert dialog.close_btn.shortcut().toString() == 'Ctrl+W'
    dialog.close_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert get_settings().application.confirm_close_window
