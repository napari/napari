from abc import ABC
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, validator

from napari.layers.utils.style_encoding import (
    ConstantStyleEncoding,
    DerivedStyleEncoding,
    DirectStyleEncoding,
    parse_kwargs_as_encoding,
)

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from .color_transformations import ColorType

DEFAULT_COLOR = transform_color('cyan')


class ConstantColorEncoding(ConstantStyleEncoding):
    """Encodes color values from a constant."""

    @validator('constant', pre=True, always=True)
    def _check_constant(cls, constant) -> np.ndarray:
        return transform_color(constant)[0]


class DirectColorEncoding(DirectStyleEncoding):
    """Encodes color values directly in an array."""

    @validator('array', pre=True, always=True)
    def _check_array(cls, array):
        return np.empty((0, 4)) if len(array) == 0 else transform_color(array)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default):
        return transform_color(default)[0]


class DerivedColorEncoding(DerivedStyleEncoding, ABC):
    def _fallback_value(self) -> np.ndarray:
        return DEFAULT_COLOR


class IdentityColorEncoding(DerivedColorEncoding):
    """Encodes color values directly from a property column.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the desired color values.
    """

    property_name: str = Field(..., allow_mutation=False)

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return transform_color(properties[self.property_name][indices])


class DiscreteColorEncoding(DerivedColorEncoding):
    """Encodes color values from a discrete property column whose values are mapped to colors.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the discrete values from which to map.
    categorical_colormap : CategoricalColormap
        Maps the selected property values to colors.
    """

    property_name: str = Field(..., allow_mutation=False)
    categorical_colormap: CategoricalColormap

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        values = properties[self.property_name][indices]
        return self.categorical_colormap.map(values)


class ContinuousColorEncoding(DerivedColorEncoding):
    """Encodes color values from a continuous property column whose values are mapped to colors.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the continuous values from which to map.
    continuous_colormap : Colormap
        Maps the selected property values to colors.
    contrast_limits : Optional[Tuple[float, float]]
        The (min, max) property values that should respectively map to the first and last
        colors in the colormap. If None, then this will attempt to calculate these values
        from the property values the first time this generate color values. If that attempt
        fails, these are effectively (0, 1).
    """

    property_name: str = Field(..., allow_mutation=False)
    continuous_colormap: Colormap
    contrast_limits: Optional[Tuple[float, float]] = None

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        all_values = properties[self.property_name]
        if self.contrast_limits is None:
            self.contrast_limits = _calculate_contrast_limits(all_values)
        values = all_values[indices]
        if self.contrast_limits is not None:
            values = np.interp(values, self.contrast_limits, (0, 1))
        return self.continuous_colormap.map(values)

    @validator('continuous_colormap', pre=True, always=True)
    def _check_continuous_colormap(
        cls, colormap: ValidColormapArg
    ) -> Colormap:
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
COLOR_ENCODINGS = (
    ContinuousColorEncoding,
    DiscreteColorEncoding,
    ConstantColorEncoding,
    IdentityColorEncoding,
    DirectColorEncoding,
)


def parse_color_encoding(
    color: Union[
        Union[COLOR_ENCODINGS], dict, ColorType, Iterable[ColorType], None
    ],
    properties: Dict[str, np.ndarray],
) -> Union[COLOR_ENCODINGS]:
    if color is None:
        return ConstantColorEncoding(constant=DEFAULT_COLOR)
    if isinstance(color, COLOR_ENCODINGS):
        return color
    if isinstance(color, dict):
        return parse_kwargs_as_encoding(COLOR_ENCODINGS, **color)
    if isinstance(color, str) and color in properties:
        return IdentityColorEncoding(property_name=color)
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
