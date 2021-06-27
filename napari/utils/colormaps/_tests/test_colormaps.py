import numpy as np
import pytest
from vispy.color import Colormap as VispyColormap

from napari.utils.colormaps import Colormap
from napari.utils.colormaps.colormap_utils import (
    _MATPLOTLIB_COLORMAP_NAMES,
    _VISPY_COLORMAPS_ORIGINAL,
    _VISPY_COLORMAPS_TRANSLATIONS,
    AVAILABLE_COLORMAPS,
    _increment_unnamed_colormap,
    ensure_colormap,
    vispy_or_mpl_colormap,
)
from napari.utils.colormaps.vendored import cm


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
    assert _increment_unnamed_colormap(names)[0] == '[unnamed colormap 2]'

    # test that named colormaps are not incremented
    named_colormap = 'perfect_colormap'
    assert (
        _increment_unnamed_colormap(names, named_colormap)[0] == named_colormap
    )


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


def test_can_accept_named_mpl_colormap():
    """Test we can accept named mpl colormap"""
    cmap_name = 'RdYlGn'
    cmap = ensure_colormap(cmap_name)
    assert isinstance(cmap, Colormap)
    assert cmap.name == cmap_name


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


def test_vispy_colormap_amount():
    """
    Test that the amount of localized vispy colormap names matches available colormaps.
    """
    for name in _VISPY_COLORMAPS_ORIGINAL:
        assert name in _VISPY_COLORMAPS_TRANSLATIONS


def test_mpl_colormap_exists():
    """Test that all localized mpl colormap names exist."""
    for name in _MATPLOTLIB_COLORMAP_NAMES:
        assert getattr(cm, name, None) is not None


def test_colormap_error_suggestion():
    """
    Test that vispy/mpl errors, when using `display_name`, suggest `name`.
    """
    name = '"twilight_shifted"'
    display_name = 'twilight shifted'
    with pytest.raises(KeyError) as excinfo:
        vispy_or_mpl_colormap(display_name)

    assert name in str(excinfo.value)

    wrong_name = 'foobar'
    with pytest.raises(KeyError) as excinfo:
        vispy_or_mpl_colormap(wrong_name)

    assert name in str(excinfo.value)
