from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Sequence, Union

import numpy as np
from pydantic import ValidationError, parse_obj_as, validator

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import Event, EventedModel
from .color_transformations import ColorType


class StyleEncoding(EventedModel, ABC):
    @abstractmethod
    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def _set_array(self, array: np.ndarray):
        pass

    def refresh(self, properties: Dict[str, np.ndarray], n_rows: int):
        indices = range(0, n_rows)
        self._set_array(self.apply(properties, indices))

    def add(self, properties: Dict[str, np.ndarray], num_to_add: int):
        num_values = len(self._get_array())
        indices = range(num_values, num_values + num_to_add)
        array = self.apply(properties, indices)
        new_array = (
            array
            if num_values == 0
            else np.append(self._get_array(), array, axis=0)
        )
        self._set_array(new_array)

    def paste(self, properties: Dict[str, np.ndarray], array: np.ndarray):
        self._set_array(np.append(self._get_array(), array, axis=0))

    def remove(self, indices: Iterable[int]):
        self._set_array(np.delete(self._get_array(), list(indices), axis=0))


class DerivedStyleEncoding(StyleEncoding, ABC):
    # TODO: consider making this a field in StyleEncoding and excluding from serialization.
    _array: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.add(array=Event)

    @property
    def array(self) -> np.ndarray:
        return self._get_array()

    def _get_array(self) -> np.ndarray:
        return self._array

    def _set_array(self, array: np.ndarray):
        self._array = array
        self.events.array()


class DirectStyleEncoding(StyleEncoding):
    array: np.ndarray
    default: np.ndarray

    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        num_values = self.array.shape[0]
        in_bound_indices = [index for index in indices if index < num_values]
        num_default = len(indices) - len(in_bound_indices)
        return (
            self.array
            if num_default == 0
            else np.append(
                self.array[in_bound_indices],
                [self.default] * num_default,
                axis=0,
            )
        )

    def _get_array(self) -> np.ndarray:
        return self.array

    def _set_array(self, array: np.ndarray):
        self.array = array


class DirectColorEncoding(DirectStyleEncoding):
    @validator('array', pre=True, always=True)
    def _check_array(cls, array):
        return np.empty((0, 4)) if len(array) == 0 else transform_color(array)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default):
        if default is None:
            default = 'cyan'
        return transform_color(default)[0]


class ConstantColorEncoding(DerivedStyleEncoding):
    constant: np.ndarray

    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.tile(self.constant, (len(indices), 1))

    @validator('constant', pre=True, always=True)
    def _check_constant(cls, constant) -> np.ndarray:
        return (
            [0, 1, 1, 1] if constant is None else transform_color(constant)[0]
        )


class IdentityColorEncoding(DerivedStyleEncoding):
    property_name: str

    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return transform_color(properties[self.property_name][indices])


class DiscreteColorEncoding(DerivedStyleEncoding):
    property_name: str
    categorical_colormap: CategoricalColormap

    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return self.categorical_colormap.map(
            properties[self.property_name][indices]
        )


class ContinuousColorEncoding(DerivedStyleEncoding):
    property_name: str
    continuous_colormap: Colormap

    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> List[ColorType]:
        return self.continuous_colormap.map(
            properties[self.property_name][indices]
        )

    @validator('continuous_colormap', pre=True, always=True)
    def _check_continuous_colormap(
        cls, colormap: ValidColormapArg
    ) -> Colormap:
        return ensure_colormap(colormap)


class FormatStringEncoding(DerivedStyleEncoding):
    format_string: str

    def apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.array(
            [
                self.format_string.format(
                    **{
                        name: column[index]
                        for name, column in properties.items()
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
        if default is None:
            default = ''
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
