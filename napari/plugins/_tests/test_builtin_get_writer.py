import os

from napari.plugins import _builtins


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_succeeds(
    napari_plugin_manager, tmpdir, layer_data_and_types
):
    """Test writing layers data."""

    _, layer_data, layer_types, filenames = layer_data_and_types
    path = os.path.join(tmpdir, 'layers_folder')

    writer = napari_plugin_manager.hook.napari_get_writer(
        path=path, layer_types=layer_types
    )

    # Write data
    assert writer == _builtins.write_layer_data_with_plugins
    assert writer(path, layer_data, plugin_name=None)

    # Check folder and files exist
    assert os.path.isdir(path)
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}
