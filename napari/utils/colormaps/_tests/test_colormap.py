import numpy as np

from napari.utils.colormaps import Colormap


def test_linear_colormap():
    """Test a linear colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    camp = Colormap(colors, name='testing')

    assert camp.name == 'testing'
    assert camp.interpolation == 'linear'
    assert len(camp.controls) == len(colors)
    np.testing.assert_almost_equal(camp.colors, colors)
    np.testing.assert_almost_equal(camp.map([0.75]), [[0, 0.5, 0.5, 1]])


def test_linear_colormap_with_control_points():
    """Test a linear colormap with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    camp = Colormap(colors, name='testing', controls=[0, 0.75, 1])

    assert camp.name == 'testing'
    assert camp.interpolation == 'linear'
    assert len(camp.controls) == len(colors)
    np.testing.assert_almost_equal(camp.colors, colors)
    np.testing.assert_almost_equal(camp.map([0.75]), [[0, 1, 0, 1]])


def test_binned_colormap():
    """Test a binned colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    camp = Colormap(colors, name='testing', interpolation='zero')

    assert camp.name == 'testing'
    assert camp.interpolation == 'zero'
    assert len(camp.controls) == len(colors) + 1
    np.testing.assert_almost_equal(camp.colors, colors)
    np.testing.assert_almost_equal(camp.map([0.4]), [[0, 1, 0, 1]])


def test_binned_colormap_with_control_points():
    """Test a binned with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    camp = Colormap(
        colors, name='testing', interpolation='zero', controls=[0, 0.2, 0.3, 1]
    )

    assert camp.name == 'testing'
    assert camp.interpolation == 'zero'
    assert len(camp.controls) == len(colors) + 1
    np.testing.assert_almost_equal(camp.colors, colors)
    np.testing.assert_almost_equal(camp.map([0.4]), [[0, 0, 1, 1]])
