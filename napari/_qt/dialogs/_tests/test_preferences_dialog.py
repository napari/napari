from napari._qt.dialogs.preferences_dialog import PreferencesDialog


def test_preferences_dialog_show(qtbot):

    dlg = PreferencesDialog()

    qtbot.addWidget(dlg)
    dlg.show()

    assert dlg.isVisible()
