import re

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
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.colormaps.vendored import cm


@pytest.mark.parametrize("name", list(AVAILABLE_COLORMAPS.keys()))
def test_colormap(name):
    if name == 'label_colormap':
        pytest.skip(
            'label_colormap is inadvertantly added to AVAILABLE_COLORMAPS but is not a normal colormap'
        )

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


def test_can_degrade_gracefully():
    """Test that we can degrade gracefully if given something not recognized."""
    with pytest.warns(UserWarning):
        cmap = ensure_colormap(object)
    assert isinstance(cmap, Colormap)
    assert cmap.name == 'gray'


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


@pytest.mark.parametrize(
    "name,display_name",
    [
        ('twilight_shifted', 'twilight shifted'),  # MPL
        ('light_blues', 'light blues'),  # Vispy
    ],
)
def test_colormap_error_suggestion(name, display_name):
    """
    Test that vispy/mpl errors, when using `display_name`, suggest `name`.
    """
    with pytest.raises(
        KeyError, match=rf"{display_name}.*you might want to use.*{name}"
    ):
        vispy_or_mpl_colormap(display_name)


def test_colormap_error_from_inexistent_name():
    """
    Test that vispy/mpl errors when using a wrong name.
    """
    name = 'foobar'
    with pytest.raises(KeyError, match=rf"{name}.*Recognized colormaps are"):
        vispy_or_mpl_colormap(name)


np.random.seed(0)
_SINGLE_RGBA_COLOR = np.random.rand(4)
_SINGLE_RGB_COLOR = _SINGLE_RGBA_COLOR[:3]
_SINGLE_COLOR_VARIANTS = (
    _SINGLE_RGB_COLOR,
    _SINGLE_RGBA_COLOR,
    tuple(_SINGLE_RGB_COLOR),
    tuple(_SINGLE_RGBA_COLOR),
    list(_SINGLE_RGB_COLOR),
    list(_SINGLE_RGBA_COLOR),
)


@pytest.mark.parametrize('color', _SINGLE_COLOR_VARIANTS)
def test_ensure_colormap_with_single_color(color):
    """See https://github.com/napari/napari/issues/3141"""
    colormap = ensure_colormap(color)
    np.testing.assert_array_equal(colormap.colors[0], [0, 0, 0, 1])
    expected_color = transform_color(color)[0]
    np.testing.assert_array_equal(colormap.colors[-1], expected_color)


np.random.seed(0)
_MULTI_RGBA_COLORS = np.random.rand(5, 4)
_MULTI_RGB_COLORS = _MULTI_RGBA_COLORS[:, :3]
_MULTI_COLORS_VARIANTS = (
    _MULTI_RGB_COLORS,
    _MULTI_RGBA_COLORS,
    tuple(tuple(color) for color in _MULTI_RGB_COLORS),
    tuple(tuple(color) for color in _MULTI_RGBA_COLORS),
    [list(color) for color in _MULTI_RGB_COLORS],
    [list(color) for color in _MULTI_RGBA_COLORS],
)


@pytest.mark.parametrize('colors', _MULTI_COLORS_VARIANTS)
def test_ensure_colormap_with_multi_colors(colors):
    """See https://github.com/napari/napari/issues/3141"""
    colormap = ensure_colormap(colors)
    expected_colors = transform_color(colors)
    np.testing.assert_array_equal(colormap.colors, expected_colors)
    assert re.match(r'\[unnamed colormap \d+\]', colormap.name) is not None


@pytest.mark.parametrize('color', ['#abc', '#abcd', '#abcdef', '#00ABCDEF'])
def test_ensure_colormap_with_hex_color_string(color):
    """
    Test all the accepted hex color representations (single/double digit rgb with/without alpha)
    """
    cmap = ensure_colormap(color)
    assert isinstance(cmap, Colormap)
    assert cmap.name == color.lower()


@pytest.mark.parametrize('color', ['#f0f', '#f0fF', '#ff00ff', '#ff00ffFF'])
def test_ensure_colormap_with_recognized_hex_color_string(color):
    """
    Test that a hex color string for magenta is associated with the existing magenta colormap
    """
    cmap = ensure_colormap(color)
    assert isinstance(cmap, Colormap)
    assert cmap.name == 'magenta'


def test_ensure_colormap_error_with_invalid_hex_color_string():
    """
    Test that ensure_colormap errors when using an invalid hex color string
    """
    color = '#ff'
    with pytest.raises(KeyError, match=rf"{color}.*Recognized colormaps are"):
        ensure_colormap(color)


@pytest.mark.parametrize('mpl_name', ['chartreuse', 'chocolate', 'lavender'])
def test_ensure_colormap_with_recognized_mpl_color_name(mpl_name):
    """
    Test that the colormap name is identical to the the mpl color name passed to ensure_colormap
    """
    cmap = ensure_colormap(mpl_name)
    assert isinstance(cmap, Colormap)
    assert cmap.name == mpl_name
