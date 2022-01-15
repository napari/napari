import os

import pytest

from napari.plugins.io import save_layers

BUILTINS = 'napari'


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_save_layer_single_named_plugin(tmpdir, layer_data_and_types):
    """Test saving a single layer with a named plugin."""
    layers, _, _, filenames = layer_data_and_types

    for layer, fn in zip(layers, filenames):
        path = os.path.join(tmpdir, fn)

        # Check file does not exist
        assert not os.path.isfile(path)

        # Write data
        save_layers(path, [layer], plugin=BUILTINS)

        # Check file now exists
        assert os.path.isfile(path)


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_save_layer_no_results(tmpdir):
    """Test no layers is not an error, and warns on no results."""

    with pytest.warns(UserWarning):
        result = save_layers('no_layers', [], plugin=BUILTINS)
        assert result == []


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_save_layer_single_no_named_plugin(tmpdir, layer_data_and_types):
    """Test saving a single layer without naming plugin."""
    # make writer builtin plugins get called first
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_write_image.bring_to_front(['builtins'])
    plugin_manager.hooks.napari_write_points.bring_to_front(['builtins'])

    layers, _, _, filenames = layer_data_and_types

    for layer, fn in zip(layers, filenames):
        path = os.path.join(tmpdir, fn)

        # Check file does not exist
        assert not os.path.isfile(path)

        # Write data
        save_layers(path, [layer])

        # Check file now exists
        assert os.path.isfile(path)


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_save_layer_multiple_named_plugin(tmpdir, layer_data_and_types):
    """Test saving multiple layers with a named plugin."""
    layers, _, _, filenames = layer_data_and_types

    path = os.path.join(tmpdir, 'layers_folder')

    # Check file does not exist
    assert not os.path.isdir(path)

    # Write data
    save_layers(path, layers, plugin=BUILTINS)

    # Check folder now exists
    assert os.path.isdir(path)

    # Check individual files now exist
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    # Check no additional files exist
    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_save_layer_multiple_no_named_plugin(tmpdir, layer_data_and_types):
    """Test saving multiple layers without naming a plugin."""
    # make writer builtin plugins get called first
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_get_writer.bring_to_front(['builtins'])

    layers, _, _, filenames = layer_data_and_types

    path = os.path.join(tmpdir, 'layers_folder')

    # Check file does not exist
    assert not os.path.isdir(path)

    # Write data
    save_layers(path, layers, plugin=BUILTINS)

    # Check folder now exists
    assert os.path.isdir(path)

    # Check individual files now exist
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    # Check no additional files exist
    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}
