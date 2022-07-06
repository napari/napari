import os

from napari.plugins._builtins import write_layer_data_with_plugins


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_succeeds(builtins, tmpdir, layer_data_and_types):
    """Test writing layers data."""
    _, layer_data, _, filenames = layer_data_and_types
    path = os.path.join(tmpdir, 'layers_folder')

    # Write data
    assert write_layer_data_with_plugins(path, layer_data, plugin_name=None)

    # Check folder and files exist
    assert os.path.isdir(path)
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}
