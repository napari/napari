from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from pydantic import Field, validator
from typing_extensions import Protocol, runtime_checkable

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from .color_transformations import ColorType
from .style_encoding import (
    ConstantStyleEncoding,
    DerivedStyleEncoding,
    DirectStyleEncoding,
    EncodingType,
    IndicesType,
    StyleEncoding,
    parse_kwargs_as_encoding,
)


class ColorValue(np.ndarray):
    """A 4x1 array that represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return transform_color(val)[0]


class ColorArray(np.ndarray):
    """An Nx4 array where each row of N represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.empty((0, 4)) if len(val) == 0 else transform_color(val)


@runtime_checkable
class ColorEncoding(StyleEncoding[ColorArray], Protocol):
    """Encodes colors from properties."""


"""The default color to use, which may also be used a safe fallback color."""
DEFAULT_COLOR = 'cyan'


class ConstantColorEncoding(ConstantStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : ColorValue
        The constant color RGBA value.
    """

    type: EncodingType = Field(
        EncodingType.CONSTANT, const=EncodingType.CONSTANT
    )
    constant: ColorValue


class DirectColorEncoding(DirectStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values directly in an array attribute.

    Attributes
    ----------
    array : ColorArray
        The array of color values. Can be written to directly to make
        persistent updates.
    default : ColorValue
        The default color value.
    """

    type: EncodingType = Field(EncodingType.DIRECT, const=EncodingType.DIRECT)
    array: ColorArray
    default: ColorValue = DEFAULT_COLOR


class IdentityColorEncoding(DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values directly from a property column.

    Attributes
    ----------
    property : str
        The name of the property that contains the desired color values.
    fallback : ColorArray
        The safe constant fallback color to use if the property column
        does not contain valid color values.
    """

    type: EncodingType = Field(
        EncodingType.IDENTITY, const=EncodingType.IDENTITY
    )
    property: str
    fallback: ColorValue = DEFAULT_COLOR

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: IndicesType
    ) -> ColorArray:
        return transform_color(properties[self.property][indices])


class NominalColorEncoding(DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a nominal property whose values are mapped to colors.

    Attributes
    ----------
    property : str
        The name of the property that contains the nominal values to be mapped to colors.
    colormap : CategoricalColormap
        Maps the property values to colors.
    fallback : ColorValue
        The safe constant fallback color to use if mapping the property values to
        colors fails.
    """

    type: EncodingType = Field(
        EncodingType.NOMINAL, const=EncodingType.NOMINAL
    )
    property: str
    colormap: CategoricalColormap
    fallback: ColorValue = DEFAULT_COLOR

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: IndicesType
    ) -> ColorArray:
        values = properties[self.property][indices]
        return self.colormap.map(values)


class QuantitativeColorEncoding(DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a quantitative property whose values are mapped to colors.

    Attributes
    ----------
    property : str
        The name of the property that contains the nominal values to be mapped to colors.
    colormap : Colormap
        Maps property values to colors.
    contrast_limits : Optional[Tuple[float, float]]
        The (min, max) property values that should respectively map to the first and last
        colors in the colormap. If None, then this will attempt to calculate these values
        from the property values the first time this generate color values. If that attempt
        fails, these are effectively (0, 1).
    fallback : ColorValue
        The safe constant fallback color to use if mapping the property values to
        colors fails.
    """

    type: EncodingType = Field(
        EncodingType.QUANTITATIVE, const=EncodingType.QUANTITATIVE
    )
    property: str
    colormap: Colormap
    contrast_limits: Optional[Tuple[float, float]] = None
    fallback: ColorValue = DEFAULT_COLOR

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: IndicesType
    ) -> ColorArray:
        all_values = properties[self.property]
        if self.contrast_limits is None:
            self.contrast_limits = _calculate_contrast_limits(all_values)
        values = all_values[indices]
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


# Define supported encodings as tuples instead of Union, so that they can be used with
# isinstance without relying on get_args, which was only added in python 3.8.

"""The color encodings supported by napari in order of precedence."""
_COLOR_ENCODINGS = (
    QuantitativeColorEncoding,
    NominalColorEncoding,
    ConstantColorEncoding,
    IdentityColorEncoding,
    DirectColorEncoding,
)


def validate_color_encoding(
    color: Union[ColorEncoding, dict, ColorType, Iterable[ColorType], None],
    properties: Dict[str, np.ndarray],
) -> ColorEncoding:
    """Validates and coerces an input to a ColorEncoding.

    Parameters
    ----------
    color : Union[ColorEncoding, dict, ColorType, Iterable[ColorType], None]
        The input or RHS of an assignment to a ColorEncoding field. If this
        is already an ColorEncoding, it is returned as is. If this is a dict,
        then we try to parse that as one of the built-in ColorEncodings. If
        this is a string and a property name, then we return an identity
        color encoding based on that property. Otherwise we try to parse the
        input as a single constant color encoding or a direct encoding of
        multiple colors.
    properties : Dict[str, np.ndarray]
        The property values, which typically come from a layer.

    Returns
    -------
    ColorEncoding

    Raises
    ------
    TypeError
        If the input is not a supported type.
    ValidationError, ValueError, AttributeError, KeyError
        If the input cannot be parsed into a color encoding.
    """
    if color is None:
        return ConstantColorEncoding(constant=DEFAULT_COLOR)
    if isinstance(color, ColorEncoding):
        return color
    if isinstance(color, dict):
        return parse_kwargs_as_encoding(_COLOR_ENCODINGS, **color)
    if isinstance(color, str) and color in properties:
        return IdentityColorEncoding(property=color)
    # TODO: distinguish between single color and array of length one as constant vs. direct.
    color_array = transform_color(color)
    if color_array.shape[0] > 1:
        return DirectColorEncoding(array=color_array, default=DEFAULT_COLOR)
    return ConstantColorEncoding(constant=color)


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
