import numpy as np
import pytest

from napari.components.experimental.chunk._commands import LoaderCommands


@pytest.mark.async_only
def test_help(make_napari_viewer, capsys):
    """Test loader.help."""
    viewer = make_napari_viewer()
    loader = loader = viewer.experimental.cmds.loader

    loader.help
    out, _ = capsys.readouterr()
    assert out.count('\n') >= 4


@pytest.mark.async_only
def test_no_layers(make_napari_viewer, capsys):
    """Test loader.layers with no layers."""
    viewer = make_napari_viewer()
    LoaderCommands(viewer.layers).layers
    out, _ = capsys.readouterr()
    for x in ["ID", "NAME", "LAYER"]:  # Just check a few.
        assert x in out


@pytest.mark.async_only
def test_one_layer(make_napari_viewer, capsys):
    """Test loader.layer with one layer."""
    viewer = make_napari_viewer()
    loader = viewer.experimental.cmds.loader

    data = np.random.random((10, 15))
    viewer.add_image(data, name="pizza")
    loader.layers
    out, _ = capsys.readouterr()
    assert "pizza" in out
    assert "(10, 15)" in out


@pytest.mark.async_only
def test_many_layers(make_napari_viewer, capsys):
    """Test loader.layer with many layers."""
    viewer = make_napari_viewer()
    loader = viewer.experimental.cmds.loader

    num_images = 10
    for i in range(num_images):
        data = np.random.random((10, 15))
        viewer.add_image(data)
    loader.layers
    out, _ = capsys.readouterr()
    assert out.count("(10, 15)") == num_images


@pytest.mark.async_only
def test_levels(make_napari_viewer, capsys):
    """Test loader.levels."""
    viewer = make_napari_viewer()
    loader = viewer.experimental.cmds.loader

    data = np.random.random((10, 15))
    viewer.add_image(data, name="pizza")
    loader.levels(0)
    out, _ = capsys.readouterr()

    # Output has color escape codes to have to check one word at a time.
    assert out.count("Levels") == 1
    assert out.count(": 1") == 1
    assert out.count("Name") == 1
    assert out.count(": pizza") == 1
    assert out.count("Shape") == 1
    assert out.count(": (10, 15)") == 1
