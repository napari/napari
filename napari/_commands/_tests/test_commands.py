import numpy as np

from napari._commands import CommandProcessor


def test_help(make_test_viewer, capsys):
    """Test cmd.help."""
    viewer = make_test_viewer()
    viewer.cmd.help
    out, _ = capsys.readouterr()
    assert out.count('\n') >= 4


def test_no_layers(make_test_viewer, capsys):
    """Test cmd.layer with no layers."""
    viewer = make_test_viewer()
    CommandProcessor(viewer.layers).layers
    out, _ = capsys.readouterr()
    for x in ["ID", "NAME", "LAYER"]:  # Just check a few.
        assert x in out


def test_one_layer(make_test_viewer, capsys):
    """Test cmd.layer with one layer."""
    viewer = make_test_viewer()
    data = np.random.random((10, 15))
    viewer.add_image(data, name="pizza")
    viewer.cmd.layers
    out, _ = capsys.readouterr()
    assert "pizza" in out
    assert "(10, 15)" in out


def test_many_layers(make_test_viewer, capsys):
    """Test cmd.layer with many layers."""
    viewer = make_test_viewer()
    num_images = 10
    for i in range(num_images):
        data = np.random.random((10, 15))
        viewer.add_image(data)
    viewer.cmd.layers
    out, _ = capsys.readouterr()
    assert out.count("(10, 15)") == num_images


def test_levels(make_test_viewer, capsys):
    """Test cmd.levels."""
    viewer = make_test_viewer()
    data = np.random.random((10, 15))
    viewer.add_image(data, name="pizza")
    viewer.cmd.levels(0)
    out, _ = capsys.readouterr()

    # Output has color escape codes to have to check in pieces
    assert out.count("Levels") == 1
    assert out.count(": 1") == 1
    assert out.count("Name") == 1
    assert out.count(": pizza") == 1
    assert out.count("Shape") == 1
    assert out.count(": (10, 15)") == 1
