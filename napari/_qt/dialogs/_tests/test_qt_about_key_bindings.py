import numpy as np

from napari._qt.dialogs.qt_about_key_bindings import QtAboutKeyBindings


def test_about_key_bindings_dialog(make_napari_viewer):
    """Test creating QtAboutKeyBindings dialog window and its methods."""
    viewer = make_napari_viewer()
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


def test_show_key_bindings_dialog(make_napari_viewer, monkeypatch):
    """Test creating dialog with method of qt_viewer."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer
    # check that dialog does not exist yet
    assert not view.findChild(QtAboutKeyBindings)
    # turn off showing the dialog for test
    monkeypatch.setattr(QtAboutKeyBindings, 'show', lambda *a: None)
    # create the dialog and make sure that it now exists
    view.show_key_bindings_dialog()
    assert isinstance(view.findChild(QtAboutKeyBindings), QtAboutKeyBindings)


def test_updating_with_layer_change(make_napari_viewer, monkeypatch):
    """Test that the dialog text updates when the active layer is changed."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer
    # turn off showing the dialog for test
    monkeypatch.setattr(QtAboutKeyBindings, 'show', lambda *a: None)
    view.show_key_bindings_dialog()
    dialog = view.findChild(QtAboutKeyBindings)

    # add an image layer
    viewer.add_image(np.random.random((5, 5, 10, 15)))
    # capture dialog text after active_layer events
    active_img_layer_text = dialog.textEditBox.toHtml()
    dialog.update_active_layer()  # force an to update to dialog
    # check that the text didn't update without a change in the active layer
    assert dialog.textEditBox.toHtml() == active_img_layer_text

    # add a shape layer (different keybindings)
    viewer.add_shapes(None, shape_type='polygon')
    # check that the new layer is the active_layer
    assert viewer.layers.selection.active == viewer.layers[1]
    # capture dialog text after active_layer events
    active_shape_layer_text = dialog.textEditBox.toHtml()
    # check that the text has changed for the new key bindings
    assert active_shape_layer_text != active_img_layer_text
    dialog.update_active_layer()  # force an update to dialog
    # check that the text didn't update without a change in the active layer
    assert dialog.textEditBox.toHtml() == active_shape_layer_text
