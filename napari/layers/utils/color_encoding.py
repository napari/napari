from abc import ABC, abstractmethod
from typing import Collection, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, validator

from napari.layers.utils.style_encoding import (
    StyleEncoding,
    _append_maybe_empty,
    _broadcast_to,
    _delete_in_bounds,
    _get_array_derived,
    _get_array_direct,
    parse_kwargs_as_encoding,
)

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from .color_transformations import ColorType

"""A 4x1 RGBA array that represents a single color value."""
ColorArray = Array[float, (4,)]

"""A Nx4 array where each row of N represents a single color value."""
MultiColorArray = Array[float, (-1, 4)]

"""The default color to use, which may also be used a safe fallback color."""
DEFAULT_COLOR = 'cyan'


class ConstantColorEncoding(EventedModel, StyleEncoding):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : ColorArray
        The constant color value.
    """

    constant: ColorArray = Field(..., allow_mutation=False)

    @validator('constant', pre=True, always=True)
    def _check_constant(cls, constant: ColorType) -> np.ndarray:
        return transform_color(constant)[0]

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional = None,
    ) -> np.ndarray:
        return _broadcast_to(self.constant, n_rows, indices)

    def _append(self, array: np.ndarray):
        pass

    def _delete(self, indices):
        pass

    def _clear(self):
        pass


class DirectColorEncoding(EventedModel, StyleEncoding):
    """Encodes color values directly in an array attribute.

    Attributes
    ----------
    array : np.ndarray
        The array of color values. Can be written to directly to make
        persistent updates.
    default : ColorArray
        The default color value.
    """

    array: MultiColorArray
    default: ColorArray = DEFAULT_COLOR

    @validator('array', pre=True, always=True)
    def _check_array(cls, array: Collection[ColorType]) -> np.ndarray:
        return _transform_maybe_empty_colors(array)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default: ColorType) -> np.ndarray:
        return transform_color(default)[0]

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional = None,
    ) -> np.ndarray:
        self.array, indexed_array = _get_array_direct(
            self.array,
            self.default,
            n_rows,
            indices,
        )
        return indexed_array

    def _append(self, array: np.ndarray):
        self.array = _append_maybe_empty(self.array, array)

    def _delete(self, indices):
        self.array = _delete_in_bounds(self.array, indices)

    def _clear(self):
        self.array = np.empty((0, 4))


class DerivedColorEncoding(EventedModel, StyleEncoding, ABC):
    """Encodes color values directly from a property column.

    Attributes
    ----------
    property : str
        The name of the property that contains the desired color values.
    fallback : ColorArray
        The safe constant fallback color to use if the property column
        does not contain valid color values.
    """

    property: str = Field(..., allow_mutation=False)
    fallback: ColorArray = DEFAULT_COLOR
    _array: MultiColorArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clear()

    @validator('fallback', pre=True, always=True)
    def _check_fallback(cls, fallback: ColorType) -> ColorArray:
        return transform_color(fallback)[0]

    @abstractmethod
    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ) -> np.ndarray:
        pass

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional = None,
    ) -> np.ndarray:
        self._array, indexed_array = _get_array_derived(
            self._apply,
            self._array,
            self.fallback,
            properties,
            n_rows,
            indices,
        )
        return indexed_array

    def _append(self, array: np.ndarray):
        self._array = _append_maybe_empty(self._array, array)

    def _delete(self, indices):
        self._array = _delete_in_bounds(self._array, indices)

    def _clear(self):
        self._array = np.empty((0, 4))


class IdentityColorEncoding(DerivedColorEncoding):
    """Encodes color values directly from a property column."""

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ) -> np.ndarray:
        return transform_color(properties[self.property][indices])


class DiscreteColorEncoding(DerivedColorEncoding):
    """Encodes color values from a discrete property column whose values are mapped to colors.

    Attributes
    ----------
    categorical_colormap : CategoricalColormap
        Maps the selected property values to colors.
    """

    categorical_colormap: CategoricalColormap

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ) -> np.ndarray:
        values = properties[self.property][indices]
        return self.categorical_colormap.map(values)


class ContinuousColorEncoding(DerivedColorEncoding):
    """Encodes color values from a continuous property column whose values are mapped to colors.

    Attributes
    ----------
    continuous_colormap : Colormap
        Maps the selected property values to colors.
    contrast_limits : Optional[Tuple[float, float]]
        The (min, max) property values that should respectively map to the first and last
        colors in the colormap. If None, then this will attempt to calculate these values
        from the property values the first time this generate color values. If that attempt
        fails, these are effectively (0, 1).
    """

    continuous_colormap: Colormap
    contrast_limits: Optional[Tuple[float, float]] = None

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        all_values = properties[self.property]
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


def _transform_maybe_empty_colors(colors: Collection[ColorType]) -> np.ndarray:
    return np.empty((0, 4)) if len(colors) == 0 else transform_color(colors)


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
