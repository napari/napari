import numpy as np
import numpy.testing as npt
import pytest

from napari.utils.colormaps.colormap_utils import (
    CoercedContrastLimits,
    _coerce_contrast_limits,
    label_colormap,
)

FIRST_COLORS = [
    [0.47058824, 0.14509805, 0.02352941, 1.0],
    [0.35686275, 0.8352941, 0.972549, 1.0],
    [0.57254905, 0.5372549, 0.9098039, 1.0],
    [0.42352942, 0.00784314, 0.75686276, 1.0],
    [0.2784314, 0.22745098, 0.62352943, 1.0],
    [0.67058825, 0.9254902, 0.5411765, 1.0],
    [0.56078434, 0.6784314, 0.69803923, 1.0],
    [0.5254902, 0.5647059, 0.6039216, 1.0],
    [0.99607843, 0.96862745, 0.10980392, 1.0],
    [0.96862745, 0.26666668, 0.23137255, 1.0],
]


@pytest.mark.parametrize(
    ('index', 'expected'), enumerate(FIRST_COLORS, start=1)
)
def test_label_colormap(index, expected):
    """Test the label colormap.

    Make sure that the default label colormap colors are identical
    to past versions, for UX consistency.
    """
    np.testing.assert_almost_equal(label_colormap(49).map(index), expected)


def test_label_colormap_exception():
    with pytest.raises(ValueError, match='num_colors must be >= 1'):
        label_colormap(0)

    with pytest.raises(ValueError, match='num_colors must be >= 1'):
        label_colormap(-1)

    with pytest.raises(
        ValueError, match=r'.*Only up to 2\*\*16=65535 colors are supported'
    ):
        label_colormap(2**16 + 1)


def test_coerce_contrast_limits_with_valid_input():
    contrast_limits = (0.0, 1.0)
    result = _coerce_contrast_limits(contrast_limits)
    assert isinstance(result, CoercedContrastLimits)
    assert np.allclose(result.contrast_limits, contrast_limits)
    assert result.offset == 0
    assert np.isclose(result.scale, 1.0)
    npt.assert_allclose(
        result.contrast_limits, result.coerce_data(np.array(contrast_limits))
    )


def test_coerce_contrast_limits_with_large_values():
    contrast_limits = (0, float(np.finfo(np.float32).max) * 100)
    result = _coerce_contrast_limits(contrast_limits)
    assert isinstance(result, CoercedContrastLimits)
    assert np.isclose(result.contrast_limits[0], np.finfo(np.float32).min / 8)
    assert np.isclose(result.contrast_limits[1], np.finfo(np.float32).max / 8)
    assert result.offset < 0
    assert result.scale < 1.0
    npt.assert_allclose(
        result.contrast_limits, result.coerce_data(np.array(contrast_limits))
    )


def test_coerce_contrast_limits_with_large_values_symmetric():
    above_float32_max = float(np.finfo(np.float32).max) * 100
    contrast_limits = (-above_float32_max, above_float32_max)
    result = _coerce_contrast_limits(contrast_limits)
    assert isinstance(result, CoercedContrastLimits)
    assert np.isclose(result.contrast_limits[0], np.finfo(np.float32).min / 8)
    assert np.isclose(result.contrast_limits[1], np.finfo(np.float32).max / 8)
    assert result.offset == 0
    assert result.scale < 1.0
    npt.assert_allclose(
        result.contrast_limits, result.coerce_data(np.array(contrast_limits))
    )


def test_coerce_contrast_limits_with_large_values_above_limit():
    f32_max = float(np.finfo(np.float32).max)
    contrast_limits = (f32_max * 10, f32_max * 100)
    result = _coerce_contrast_limits(contrast_limits)
    assert isinstance(result, CoercedContrastLimits)
    assert np.isclose(result.contrast_limits[0], np.finfo(np.float32).min / 8)
    assert np.isclose(result.contrast_limits[1], np.finfo(np.float32).max / 8)
    assert result.offset < 0
    assert result.scale < 1.0
    npt.assert_allclose(
        result.contrast_limits, result.coerce_data(np.array(contrast_limits))
    )


def test_coerce_contrast_limits_small_values():
    contrast_limits = (1e-45, 9e-45)
    result = _coerce_contrast_limits(contrast_limits)
    assert isinstance(result, CoercedContrastLimits)
    assert np.isclose(result.contrast_limits[0], 0)
    assert np.isclose(result.contrast_limits[1], 1000)
    assert result.offset < 0
    assert result.scale > 1
    npt.assert_allclose(
        result.contrast_limits, result.coerce_data(np.array(contrast_limits))
    )
