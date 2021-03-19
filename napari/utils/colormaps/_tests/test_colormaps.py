import numpy as np
import pytest
from vispy.color import Colormap as VispyColormap

from napari.utils.colormaps import Colormap
from napari.utils.colormaps.colormap_utils import (
    AVAILABLE_COLORMAPS,
    ensure_colormap,
    increment_unnamed_colormap,
)


@pytest.mark.parametrize("name", list(AVAILABLE_COLORMAPS.keys()))
def test_colormap(name):
    np.random.seed(0)

    cmap = AVAILABLE_COLORMAPS[name]

    # Test can map random 0-1 values
    values = np.random.rand(50)
    colors = cmap.map(values)
    assert colors.shape == (len(values), 4)

    # Create vispy colormap and check current colormaps match vispy
    # colormap
    vispy_cmap = VispyColormap(*cmap)
    vispy_colors = vispy_cmap.map(values)
    np.testing.assert_almost_equal(colors, vispy_colors, decimal=6)


def test_increment_unnamed_colormap():
    # test that unnamed colormaps are incremented
    names = [
        '[unnamed colormap 0]',
        'existing_colormap',
        'perceptually_uniform',
        '[unnamed colormap 1]',
    ]
    assert increment_unnamed_colormap(names) == '[unnamed colormap 2]'

    # test that named colormaps are not incremented
    named_colormap = 'perfect_colormap'
    assert increment_unnamed_colormap(names, named_colormap) == named_colormap


def test_can_accept_vispy_colormaps():
    """Test that we can accept vispy colormaps."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    vispy_cmap = VispyColormap(colors)
    cmap = ensure_colormap(vispy_cmap)
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors)


def test_can_accept_napari_colormaps():
    """Test that we can accept napari colormaps."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    napari_cmap = Colormap(colors)
    cmap = ensure_colormap(napari_cmap)
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors)


def test_can_accept_vispy_colormap_name_tuple():
    """Test that we can accept vispy colormap named type."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    vispy_cmap = VispyColormap(colors)
    cmap = ensure_colormap(('special_name', vispy_cmap))
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors)
    assert cmap.name == 'special_name'


def test_can_accept_napari_colormap_name_tuple():
    """Test that we can accept napari colormap named type."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    napari_cmap = Colormap(colors)
    cmap = ensure_colormap(('special_name', napari_cmap))
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors)
    assert cmap.name == 'special_name'


def test_can_accept_named_vispy_colormaps():
    """Test that we can accept named vispy colormap."""
    cmap = ensure_colormap('red')
    assert isinstance(cmap, Colormap)
    assert cmap.name == 'red'


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_can_accept_vispy_colormaps_in_dict():
    """Test that we can accept vispy colormaps in a dictionary."""
    colors_a = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    colors_b = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]])
    vispy_cmap_a = VispyColormap(colors_a)
    vispy_cmap_b = VispyColormap(colors_b)
    cmap = ensure_colormap({'a': vispy_cmap_a, 'b': vispy_cmap_b})
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors_a)
    assert cmap.name == 'a'


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_can_accept_napari_colormaps_in_dict():
    """Test that we can accept vispy colormaps in a dictionary"""
    colors_a = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    colors_b = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]])
    napari_cmap_a = Colormap(colors_a)
    napari_cmap_b = Colormap(colors_b)
    cmap = ensure_colormap({'a': napari_cmap_a, 'b': napari_cmap_b})
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors_a)
    assert cmap.name == 'a'


def test_can_accept_colormap_dict():
    """Test that we can accept vispy colormaps in a dictionary"""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = ensure_colormap({'colors': colors, 'name': 'special_name'})
    assert isinstance(cmap, Colormap)
    np.testing.assert_almost_equal(cmap.colors, colors)
    assert cmap.name == 'special_name'


def test_border_coords():
    """Test if borders are properly handled and Vispy Colormap is properly created"""
    coords = [0.1, 0.5, 1]
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    coords1 = [0, 0.5, 0.9]
    with pytest.warns(
        RuntimeWarning,
        match="colormap need to have first coord equal to 0, not 0.1",
    ):
        cmap = Colormap(colors=colors, controls=coords, name="test")
    assert len(cmap.controls) == 4
    with pytest.warns(
        RuntimeWarning,
        match="colormap need to have last coord equal to 1, not 0.9",
    ):
        cmap = Colormap(colors=colors, controls=coords1, name="test")
    assert len(cmap.controls) == 4
    coords2 = [0.1, 0.5, 0.9]
    with pytest.warns(RuntimeWarning) as records:
        cmap = Colormap(colors=colors, controls=coords2, name="test")
    assert len(records) == 2
    assert len(cmap.controls) == 5
    VispyColormap(*cmap)


def test_ascending_order():
    controls = [0.2, 0.1, 1]
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(
        ValueError, match="Coords needs to be sorted in ascending order"
    ):
        Colormap(colors=colors, controls=controls, name="test")

    controls = [-0.2, 0.1, 1]
    with pytest.warns(RuntimeWarning) as records:
        with pytest.raises(ValueError) as exec_info:
            Colormap(colors=colors, controls=controls, name="test")
    assert len(records) == 1
    assert (
        exec_info.value.args[0][0]
        .exc.args[0]
        .endswith("Coords needs to be in range [0, 1]")
    )
