import numpy as np
import pytest

from napari.utils.colormaps import Colormap


def test_linear_colormap():
    """Test a linear colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing')

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'linear'
    assert len(cmap.controls) == len(colors)
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.75]), [[0, 0.5, 0.5, 1]])


def test_linear_colormap_with_control_points():
    """Test a linear colormap with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing', controls=[0, 0.75, 1])

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'linear'
    assert len(cmap.controls) == len(colors)
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.75]), [[0, 1, 0, 1]])


def test_non_ascending_control_points():
    """Test non ascending control points raises an error."""
    colors = np.array(
        [[0, 0, 0, 1], [0, 0.5, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 0.25, 1])


def test_wrong_number_control_points():
    """Test wrong number of control points raises an error."""
    colors = np.array(
        [[0, 0, 0, 1], [0, 0.5, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 1])


def test_wrong_start_control_point():
    """Test wrong start of control points raises an error."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0.1, 0.75, 1])


def test_wrong_end_control_point():
    """Test wrong end of control points raises an error."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 0.9])


def test_binned_colormap():
    """Test a binned colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing', interpolation='zero')

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 1, 0, 1]])


def test_binned_colormap_with_control_points():
    """Test a binned with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(
        colors,
        name='testing',
        interpolation='zero',
        controls=[0, 0.2, 0.3, 1],
    )

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 0, 1, 1]])


def test_colormap_equality():
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap_1 = Colormap(colors, name='testing', controls=[0, 0.75, 1])
    cmap_2 = Colormap(colors, name='testing', controls=[0, 0.75, 1])
    cmap_3 = Colormap(colors, name='testing', controls=[0, 0.25, 1])
    assert cmap_1 == cmap_2
    assert cmap_1 != cmap_3
