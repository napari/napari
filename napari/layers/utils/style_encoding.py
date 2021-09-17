from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterable, List, TypeVar, Union

import numpy as np
from pydantic import ValidationError, parse_obj_as, validator

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from .color_transformations import ColorType

OutputType = TypeVar('OutputType')


class StyleStore(EventedModel, Generic[OutputType], ABC):
    array: Array = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.events.array.blocker():
            self.array = self._values_to_array([])

    def update(self, values: List[OutputType]):
        self.array = self._values_to_array(values)

    def append(self, values: List[OutputType]):
        array = self._values_to_array(values)
        self.array = np.concatenate((self.array, array), axis=0)

    def delete(self, indices: Iterable[int]):
        self.array = np.delete(self.array, list(indices), axis=0)

    def _values_to_array(self, values: List[OutputType]) -> np.ndarray:
        return np.array(values)


class ColorStore(StyleStore[ColorType]):
    def _values_to_array(self, values: List[ColorType]) -> np.ndarray:
        return transform_color(values) if len(values) > 0 else np.empty((0, 4))


class StringStore(StyleStore[str]):
    def _values_to_array(self, values: List[str]) -> np.ndarray:
        return np.array(values, dtype=str)


class StyleEncoding(EventedModel, Generic[OutputType], ABC):
    _store: StyleStore[OutputType]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._store = self._make_store()

    @abstractmethod
    def apply_to_row(
        self, property_row: Dict[str, Any], index: int
    ) -> OutputType:
        pass

    @abstractmethod
    def _make_store(self) -> StyleStore[OutputType]:
        pass

    def get_array(self) -> np.ndarray:
        return self._store.array

    def connect(self, callback):
        self._store.events.array.connect(callback)

    def refresh(self, properties: Dict[str, np.ndarray], n_rows: int):
        indices = range(0, n_rows)
        values = self._apply_to_table(properties, indices)
        self._store.update(values)

    def add(self, properties: Dict[str, np.ndarray], num_to_add: int):
        num_values = len(self._store.array)
        indices = range(num_values, num_values + num_to_add)
        values = self._apply_to_table(properties, indices)
        self._store.append(values)

    def remove(self, indices: Iterable[int]):
        self._store.delete(indices)

    def _apply_to_table(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ) -> List[OutputType]:
        return [
            self.apply_to_row(
                {name: column[index] for name, column in properties.items()},
                index,
            )
            for index in indices
        ]


# TODO: if there are no property columns, this will return 0 even if there are some data.
def _num_rows(properties: Dict[str, np.ndarray]) -> int:
    return len(next(iter(properties.values()))) if len(properties) > 0 else 0


class ColorEncodingBase(StyleEncoding[ColorType], ABC):
    def _make_store(self):
        return ColorStore()


class DirectColorEncoding(ColorEncodingBase):
    values: List[ColorType]
    default_value: ColorType

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.values.connect(self._on_values_changed)
        self._on_values_changed()

    def _on_values_changed(self, event=None):
        self._store.update(self.values)

    def apply_to_row(
        self, property_row: Dict[str, Any], index: int
    ) -> ColorType:
        return (
            self.values[index]
            if index < len(self.values)
            else self.default_value
        )

    def remove(self, indices):
        super().remove(indices)
        for index in sorted(indices, reverse=True):
            self.values.pop(index)


DirectColorEncoding.__eq_operators__['default_value'] = np.array_equal


class ConstantColorEncoding(ColorEncodingBase):
    constant: ColorType

    def apply_to_row(
        self, property_row: Dict[str, Any], index: int
    ) -> ColorType:
        return self.constant


ConstantColorEncoding.__eq_operators__['constant'] = np.array_equal


class IdentityColorEncoding(ColorEncodingBase):
    property_name: str

    def apply_to_row(
        self, property_row: Dict[str, Any], index: int
    ) -> ColorType:
        return property_row[self.property_name]


class DiscreteColorEncoding(ColorEncodingBase):
    property_name: str
    categorical_colormap: CategoricalColormap

    def apply_to_row(
        self, property_row: Dict[str, Any], index: int
    ) -> ColorType:
        return self.categorical_colormap.map(property_row[self.property_name])[
            0
        ]


class ContinuousColorEncoding(ColorEncodingBase):
    property_name: str
    continuous_colormap: Colormap

    def apply_to_row(
        self, property_row: Dict[str, Any], index: int
    ) -> ColorType:
        return self.continuous_colormap.map(property_row[self.property_name])[
            0
        ]

    @validator('continuous_colormap', pre=True, always=True)
    def _check_continuous_colormap(
        cls, colormap: ValidColormapArg
    ) -> Colormap:
        return ensure_colormap(colormap)


class StringEncodingBase(StyleEncoding[str], ABC):
    def _make_store(self):
        return StringStore()


class FormatStringEncoding(StringEncodingBase):
    format_string: str

    def apply_to_row(self, property_row: Dict[str, Any], index: int) -> str:
        return self.format_string.format(**property_row)


class DirectStringEncoding(StringEncodingBase):
    values: List[str]
    default_value: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.values.connect(self._on_values_changed)
        self._on_values_changed()

    def _on_values_changed(self, event=None):
        self._store.update(self.values)

    def apply_to_row(self, property_row: Dict[str, Any], index: int) -> str:
        return (
            self.values[index]
            if index < len(self.values)
            else self.default_value
        )

    def remove(self, indices):
        super().remove(indices)
        for index in sorted(indices, reverse=True):
            self.values.pop(index)


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
