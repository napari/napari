from napari._qt.dialogs.qt_about_key_bindings import QtAboutKeyBindings


def test_about_key_bindings_dialog(make_test_viewer):
    """Test creating QtAboutKeyBindings dialog window and its methods."""
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer
    key_bindings_dialog = QtAboutKeyBindings(viewer, view._key_map_handler)
    assert key_bindings_dialog.viewer == viewer
    assert key_bindings_dialog.key_map_handler == view._key_map_handler

    # check ability to update the active layer, should've ran during init too
    key_bindings_dialog.update_active_layer()

    # check ability to change layer type selected in dropdown
    for layer in key_bindings_dialog.key_bindings_strs.keys():
        key_bindings_dialog.change_layer_type(layer)
        # Could test to make sure text has been updated, but `toHtml` does not
        # return the same str that is provided to `setHtml`, so this could get
        # messy...


def test_show_key_bindings_dialog(make_test_viewer, monkeypatch):
    """Test creating dialog with method of qt_viewer."""
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer
    # check that dialog does not exist yet
    assert not view.findChild(QtAboutKeyBindings)
    # turn off showing the dialog for test
    monkeypatch.setattr(QtAboutKeyBindings, 'show', lambda *a: None)
    # create the dialog and make sure that it now exists
    view.show_key_bindings_dialog()
    assert isinstance(view.findChild(QtAboutKeyBindings), QtAboutKeyBindings)
