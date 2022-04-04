from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import parse_obj_as, validator
from typing_extensions import Protocol, runtime_checkable

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.translations import trans
from .color_transformations import ColorType
from .style_encoding import (
    StyleEncoding,
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
)


class ColorValue(np.ndarray):
    """A 4x1 array that represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return transform_color(val)[0]

    def __eq__(self, other) -> bool:
        return np.all(self == other)


class ColorArray(np.ndarray):
    """An Nx4 array where each row of N represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return (
            np.empty((0, 4), np.float32)
            if len(val) == 0
            else transform_color(val)
        )

    def __eq__(self, other) -> bool:
        return np.all(self == other)


@runtime_checkable
class ColorEncoding(StyleEncoding[ColorValue, ColorArray], Protocol):
    """Encodes colors from features."""


"""The default color to use, which may also be used a safe fallback color."""
# DEFAULT_COLOR = ColorValue.validate_type('cyan')
DEFAULT_COLOR = 'cyan'


class ConstantColorEncoding(_ConstantStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : ColorValue
        The constant color RGBA value.
    """

    encoding_type: Literal['ConstantColorEncoding'] = 'ConstantColorEncoding'
    constant: ColorValue


class ManualColorEncoding(_ManualStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values manually in an array attribute.

    Attributes
    ----------
    array : ColorArray
        The array of color values. Can be written to directly to make
        persistent updates.
    default : ColorValue
        The default color value.
    """

    encoding_type: Literal['ManualColorEncoding'] = 'ManualColorEncoding'
    array: ColorArray
    default: ColorValue = DEFAULT_COLOR


class DirectColorEncoding(_DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values directly from a nominal feature column.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the desired color values.
    fallback : ColorArray
        The safe constant fallback color to use if the feature column
        does not contain valid color values.
    """

    encoding_type: Literal['DirectColorEncoding'] = 'DirectColorEncoding'
    feature: str
    fallback: ColorValue = DEFAULT_COLOR

    def __call__(self, features: Any) -> ColorArray:
        return ColorArray.validate_type(np.asarray(features[self.feature]))


class NominalColorEncoding(_DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a nominal feature whose values are mapped to colors.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the nominal values to be mapped to colors.
    colormap : CategoricalColormap
        Maps the feature values to colors.
    fallback : ColorValue
        The safe constant fallback color to use if mapping the feature values to
        colors fails.
    """

    encoding_type: Literal['NominalColorEncoding'] = 'NominalColorEncoding'
    feature: str
    colormap: CategoricalColormap
    fallback: ColorValue = DEFAULT_COLOR

    def _apply(self, features: Any) -> ColorArray:
        values = features[self.feature]
        return self.colormap.map(values)


class QuantitativeColorEncoding(_DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a quantitative feature whose values are mapped to colors.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the nominal values to be mapped to colors.
    colormap : Colormap
        Maps feature values to colors.
    contrast_limits : Optional[Tuple[float, float]]
        The (min, max) feature values that should respectively map to the first and last
        colors in the colormap. If None, then this will attempt to calculate these values
        from the feature values the first time this generate color values. If that attempt
        fails, these are effectively (0, 1).
    fallback : ColorValue
        The safe constant fallback color to use if mapping the feature values to
        colors fails.
    """

    encoding_type: Literal[
        'QuantitativeColorEncoding'
    ] = 'QuantitativeColorEncoding'
    feature: str
    colormap: Colormap
    contrast_limits: Optional[Tuple[float, float]] = None
    fallback: ColorValue = DEFAULT_COLOR

    def _apply(self, features: Any) -> ColorArray:
        values = features[self.feature]
        if self.contrast_limits is None:
            self.contrast_limits = _calculate_contrast_limits(values)
        if self.contrast_limits is not None:
            values = np.interp(values, self.contrast_limits, (0, 1))
        return self.colormap.map(values)

    @validator('colormap', pre=True, always=True)
    def _check_colormap(cls, colormap: ValidColormapArg) -> Colormap:
        return ensure_colormap(colormap)

    @validator('contrast_limits', pre=True, always=True)
    def _check_contrast_limits(
        cls, contrast_limits
    ) -> Optional[Tuple[float, float]]:
        if (contrast_limits is not None) and (
            contrast_limits[0] >= contrast_limits[1]
        ):
            raise ValueError(
                'contrast_limits must be a strictly increasing pair of values'
            )
        return contrast_limits


"""The types of arguments supported when setting a ColorEncoding field."""
ColorEncodingArgument = Union[ColorEncoding, dict, str, ColorType, None]


def validate_color_encoding(value: ColorEncodingArgument) -> ColorEncoding:
    """Validates and coerces a value to a StringEncoding.

    Parameters
    ----------
    value : ColorEncodingArgument
        The value to validate and coerce.
        If this is already a StringEncoding, it is returned as is.
        If this is a dict, then it should represent one of the built-in string encodings.
        If this a valid format string, then a FormatStringEncoding is returned.
        If this is any other string, a DirectStringEncoding is returned.
        If this is a sequence of strings, a ManualStringEncoding is returned.

    Returns
    -------
    ColorEncoding

    Raises
    ------
    TypeError
        If the value is not a supported type.
    ValidationError
        If the value cannot be parsed into a ColorEncoding.
    """
    if value is None:
        return ConstantColorEncoding(constant=DEFAULT_COLOR)
    if isinstance(value, ColorEncoding):
        return value
    if isinstance(value, dict):
        return parse_obj_as(
            Union[
                ConstantColorEncoding,
                ManualColorEncoding,
                DirectColorEncoding,
                NominalColorEncoding,
                QuantitativeColorEncoding,
            ],
            value,
        )
    if isinstance(value, str):
        # Should we guess a nominal/quantitative encoding instead?
        return DirectColorEncoding(feature=value, fallback=DEFAULT_COLOR)
    try:
        color_array = transform_color(value)
    except (ValueError, AttributeError, KeyError):
        raise TypeError(
            trans._(
                'value should be a ColorEncoding, a dict, a string, a color, a sequence of colors, or None',
                deferred=True,
            )
        )
    # TODO: distinguish between single color and array of length one as constant vs. direct.
    if color_array.shape[0] > 1:
        return ManualColorEncoding(array=color_array, default=DEFAULT_COLOR)
    return ConstantColorEncoding(constant=value)


def _calculate_contrast_limits(
    values: np.ndarray,
) -> Optional[Tuple[float, float]]:
    contrast_limits = None
    if values.size > 0:
        min_value = np.min(values)
        max_value = np.max(values)
        # Use < instead of != to handle nans.
        if min_value < max_value:
            contrast_limits = (min_value, max_value)
    return contrast_limits
