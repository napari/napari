from qtpy.QtWidgets import QDialog

from napari._qt.dialogs.shimmed_plugin_dialog import ShimmedPluginDialog
from napari.settings import get_settings
from napari.settings._constants import PluginShimWarningLevel


def test_dialog_accept(qtbot):
    dialog = ShimmedPluginDialog(None, plugins={'plugin1', 'plugin2'})
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == 'Installed Plugin Warning'
    assert dialog.plugin_text == '\n'.join({'plugin1', 'plugin2'})
    assert not dialog.only_new_checkbox.isChecked()
    dialog.okay_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert (
        get_settings().plugins.shimmed_plugin_warning_level
        == PluginShimWarningLevel.ALWAYS
    )


def test_dialog_accept_checked(qtbot):
    dialog = ShimmedPluginDialog(None, plugins={'plugin1', 'plugin2'})
    qtbot.addWidget(dialog)
    assert dialog.plugin_text == '\n'.join({'plugin1', 'plugin2'})
    assert not dialog.only_new_checkbox.isChecked()
    dialog.only_new_checkbox.setChecked(True)
    dialog.okay_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert (
        get_settings().plugins.shimmed_plugin_warning_level
        == PluginShimWarningLevel.NEW
    )
    assert get_settings().plugins.already_warned_shimmed_plugins == {
        'plugin1',
        'plugin2',
    }


def test_dialog_reject(qtbot):
    dialog = ShimmedPluginDialog(None, plugins={'plugin1', 'plugin2'})
    qtbot.addWidget(dialog)
    assert dialog.plugin_text == '\n'.join({'plugin1', 'plugin2'})
    dialog.only_new_checkbox.setChecked(True)
    dialog.reject()
    # rejected doesn't save setting
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert (
        get_settings().plugins.shimmed_plugin_warning_level
        == PluginShimWarningLevel.ALWAYS
    )
    assert get_settings().plugins.already_warned_shimmed_plugins == set()
