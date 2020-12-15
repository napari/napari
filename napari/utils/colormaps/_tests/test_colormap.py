import numpy as np
import pytest

from napari.utils.colormaps import CategoricalColormap, Colormap


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
        colors, name='testing', interpolation='zero', controls=[0, 0.2, 0.3, 1]
    )

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 0, 1, 1]])


def test_categorical_colormap_direct():
    """Test a categorical colormap with a provided mapping"""
    colormap = {'hi': np.array([1, 1, 1, 1]), 'hello': np.array([0, 0, 0, 0])}
    cmap = CategoricalColormap(colormap=colormap)

    color = cmap.map(['hi'])
    np.testing.assert_allclose(color, [[1, 1, 1, 1]])
    color = cmap.map(['hello'])
    np.testing.assert_allclose(color, [[0, 0, 0, 0]])

    # test that the default fallback color (black) is applied
    new_color_0 = cmap.map(['not a key'])
    np.testing.assert_almost_equal(new_color_0, [[0, 0, 0, 1]])
    new_cmap = cmap.colormap
    np.testing.assert_almost_equal(new_cmap['not a key'], [0, 0, 0, 1])

    # set a cycle of fallback colors
    new_fallback_colors = [[1, 0, 0, 1], [0, 1, 0, 1]]
    cmap.fallback_color = new_fallback_colors
    new_color_1 = cmap.map(['new_prop 1'])
    np.testing.assert_almost_equal(
        np.squeeze(new_color_1), new_fallback_colors[0]
    )
    new_color_2 = cmap.map(['new_prop 2'])
    np.testing.assert_almost_equal(
        np.squeeze(new_color_2), new_fallback_colors[1]
    )


def test_categorical_colormap_cycle():
    color_cycle = [[1, 1, 1, 1], [1, 0, 0, 1]]
    cmap = CategoricalColormap(color_cycle)

    # verify that no mapping between prop value and color has been set
    assert cmap.colormap == {}

    # the values used to create the color cycle can be accessed via fallback colro
    np.testing.assert_almost_equal(cmap.fallback_color, color_cycle)

    # map 2 colors, verify their colors are returned in order
    colors = cmap.map(['hi', 'hello'])
    np.testing.assert_almost_equal(colors, color_cycle)

    # map a third color and verify the colors wrap around
    third_color = cmap.map(['bonjour'])
    np.testing.assert_almost_equal(np.squeeze(third_color), color_cycle[0])


def test_categorical_with_invalid_colormap():
    with pytest.raises(TypeError):
        CategoricalColormap('viridis')
