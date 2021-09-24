from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import ValidationError, parse_obj_as, validator

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel


class StyleEncoding(EventedModel, ABC):
    array: np.ndarray = []

    @abstractmethod
    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        pass

    def update_all(self, properties: Dict[str, np.ndarray], n_rows: int):
        indices = range(0, n_rows)
        self.array = self._apply(properties, indices)

    def update_tail(self, properties: Dict[str, np.ndarray], n_rows: int):
        n_values = self.array.shape[0]
        indices = range(n_values, n_rows)
        array = self._apply(properties, indices)
        self.append(array)

    def append(self, array: np.ndarray):
        self.array = _append_maybe_empty(self.array, array)

    def delete(self, indices: Iterable[int]):
        self.array = np.delete(self.array, list(indices), axis=0)

    @validator('array', pre=True, always=True)
    def _check_array(cls, array):
        return np.array(array)


class DerivedStyleEncoding(StyleEncoding, ABC):
    def json(self, **kwargs):
        return super().json(exclude={'array'})

    def dict(self, **kwargs):
        return super().dict(exclude={'array'})


class DirectStyleEncoding(StyleEncoding):
    default: np.ndarray

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        n_values = self.array.shape[0]
        in_bound_indices = [index for index in indices if index < n_values]
        n_default = len(indices) - len(in_bound_indices)
        return _append_maybe_empty(
            self.array[in_bound_indices],
            np.array([self.default] * n_default),
        )


class DirectColorEncoding(DirectStyleEncoding):
    @validator('array', pre=True, always=True)
    def _check_array(cls, array):
        return np.empty((0, 4)) if len(array) == 0 else transform_color(array)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default):
        return transform_color(default)[0]


class ConstantColorEncoding(DerivedStyleEncoding):
    constant: np.ndarray

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.tile(self.constant, (len(indices), 1))

    @validator('constant', pre=True, always=True)
    def _check_constant(cls, constant) -> np.ndarray:
        return transform_color(constant)[0]


class IdentityColorEncoding(DerivedStyleEncoding):
    property_name: str

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return transform_color(properties[self.property_name][indices])


class DiscreteColorEncoding(DerivedStyleEncoding):
    property_name: str
    categorical_colormap: CategoricalColormap

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        values = properties[self.property_name][indices]
        return self.categorical_colormap.map(values)


class ContinuousColorEncoding(DerivedStyleEncoding):
    property_name: str
    continuous_colormap: Colormap
    contrast_limits: Optional[Tuple[float, float]] = None

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        all_values = properties[self.property_name]
        if self.contrast_limits is None:
            self.contrast_limits = self._calculate_contrast_limits(all_values)
        values = all_values[indices]
        if self.contrast_limits is not None:
            values = np.interp(values, self.contrast_limits, (0, 1))
        return self.continuous_colormap.map(values)

    @classmethod
    def _calculate_contrast_limits(
        cls, values: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        contrast_limits = None
        if values.size > 0:
            min_value = np.min(values)
            max_value = np.max(values)
            # Use < instead of != to handle nans.
            if min_value < max_value:
                contrast_limits = (min_value, max_value)
        return contrast_limits

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


class FormatStringEncoding(DerivedStyleEncoding):
    format_string: str

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.array(
            [
                self.format_string.format(
                    **{
                        name: values[index]
                        for name, values in properties.items()
                    }
                )
                for index in indices
            ]
        )


class DirectStringEncoding(DirectStyleEncoding):
    @validator('array', pre=True, always=True)
    def _check_array(cls, array) -> np.ndarray:
        return np.array(array, dtype=str)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default) -> np.ndarray:
        return np.array(default, dtype=str)


def parse_obj_as_union(union, obj: Dict[str, Any]):
    try:
        return parse_obj_as(union, obj)
    except ValidationError as error:
        raise ValueError(
            'Failed to parse a supported encoding from kwargs:\n'
            f'{obj}\n\n'
            'The kwargs must specify the fields of exactly one of the following encodings:\n'
            f'{union}\n\n'
            'Original error:\n'
            f'{error}'
        )


ColorEncoding = Union[
    ContinuousColorEncoding,
    DiscreteColorEncoding,
    ConstantColorEncoding,
    IdentityColorEncoding,
    DirectColorEncoding,
]


StringEncoding = Union[
    FormatStringEncoding,
    DirectStringEncoding,
]


def _append_maybe_empty(left: np.ndarray, right: np.ndarray):
    """Like numpy.append, except that the dimensionality of empty arrays is ignored."""
    if right.size == 0:
        return left
    if left.size == 0:
        return right
    return np.append(left, right, axis=0)
