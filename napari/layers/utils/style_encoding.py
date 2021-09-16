from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterable, List, TypeVar, Union

import numpy as np
from pydantic import validator

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel, EventEmitter
from .color_transformations import ColorType

OutputType = TypeVar('OutputType')


class StyleEncoding(EventedModel, Generic[OutputType], ABC):
    _array: np.ndarray
    _array_update: EventEmitter

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._array = self._values_to_array([])
        self._array_update = EventEmitter(source=self)

    @abstractmethod
    def apply_to_row(self, property_row: Dict[str, Any]) -> OutputType:
        pass

    def get_array(self):
        return self._array

    def connect(self, callback):
        self._array_update = callback

    def _values_to_array(self, values: List[OutputType]) -> np.ndarray:
        return np.array(values)

    def refresh(self, properties: Dict[str, np.ndarray]):
        num_values = _num_rows(properties)
        indices = range(0, num_values)
        values = self._apply_to_table(properties, indices)
        self._array = self._values_to_array(values)

    def add(self, properties: Dict[str, np.ndarray], num_to_add: int):
        num_values = _num_rows(properties)
        indices = range(num_values - num_to_add, num_values)
        values = self._apply_to_table(properties, indices)
        array = self._values_to_array(values)
        self._array = np.concatenate((self._array, array), axis=0)

    def remove(self, indices: Iterable[int]):
        self._array = np.delete(self._array, list(indices), axis=0)

    def _apply_to_table(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ) -> List[OutputType]:
        return [
            self.apply_to_row(
                {name: column[index] for name, column in properties.items()}
            )
            for index in indices
        ]


# TODO: if there are no property columns, this will return 0 even if there are some data.
def _num_rows(properties: Dict[str, np.ndarray]) -> int:
    return len(next(iter(properties.values()))) if len(properties) > 0 else 0


class ColorEncodingBase(StyleEncoding[ColorType], ABC):
    def _values_to_array(self, values: List[ColorType]) -> np.ndarray:
        return transform_color(values) if len(values) > 0 else np.empty((0, 4))


class DirectColorEncoding(ColorEncodingBase):
    values: List[ColorType]
    default_value: ColorType

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.values.connect(self._on_values_changed)
        self._on_values_changed()

    def _on_values_changed(self, event=None):
        self._array = self._values_to_array(self.values)

    def apply_to_row(self, property_row: Dict[str, Any]) -> ColorType:
        # TODO: consider returning value if we have access to row index.
        return self.default_value

    def refresh(self, properties: Dict[str, np.ndarray]):
        # TODO: should probably resize based on number of rows but can't rely on that yet.
        pass


DirectColorEncoding.__eq_operators__['default_value'] = np.array_equal


class ConstantColorEncoding(ColorEncodingBase):
    """Maps from a property row to a constant color.

    Attributes
    ----------
    constant : ColorType
        The constant value to always return.
    """

    constant: ColorType

    def apply_to_row(self, property_row: Dict[str, Any]) -> ColorType:
        return self.constant


ConstantColorEncoding.__eq_operators__['constant'] = np.array_equal


class IdentityColorEncoding(ColorEncodingBase):
    """Maps from a property row to a property value by name.

    Attributes
    ----------
    property_name : str
        The name of the property to select from a row.
    """

    property_name: str

    def apply_to_row(self, property_row: Dict[str, Any]) -> ColorType:
        return property_row[self.property_name]


class DiscreteColorEncoding(ColorEncodingBase):
    property_name: str
    mapping: CategoricalColormap

    def apply_to_row(self, property_row: Dict[str, Any]) -> ColorType:
        return self.mapping.map(property_row[self.property_name])[0]


class ContinuousColorEncoding(ColorEncodingBase):
    """Maps from a property row to a property value by name to another value defined by a discrete continuous colormap.

    Attributes
    ----------
    property_name : str
        The name of the property to select from a row.
    colormap : Colormap
        The map from the continuous named property value to the output color value.
    """

    property_name: str
    colormap: Colormap

    def apply_to_row(self, property_row: Dict[str, Any]) -> ColorType:
        return self.colormap.map(property_row[self.property_name])[0]

    @validator('colormap', pre=True, always=True)
    def _check_colormap(cls, colormap: ValidColormapArg) -> Colormap:
        return ensure_colormap(colormap)


class StringEncodingBase(StyleEncoding[str], ABC):
    def _values_to_array(self, values: List[str]) -> np.ndarray:
        return np.array(values, dtype=str)


class FormatStringEncoding(StringEncodingBase):
    """Maps from a property row to a formatted string containing property names.

    Attributes
    ----------
    format_string : str
        The format string as described in str.format. The format placeholders
        should only reference property names.
    """

    format_string: str

    def apply_to_row(self, property_row: Dict[str, Any]) -> str:
        return self.format_string.format(**property_row)


class DirectStringEncoding(StringEncodingBase):
    values: List[str]
    default_value: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.values.connect(self._on_values_changed)
        self._on_values_changed()

    def _on_values_changed(self, event=None):
        self._array = self._values_to_array(self.values)

    def apply_to_row(self, property_row: Dict[str, Any]) -> str:
        # TODO: consider returning value if we have access to row index.
        return self.default_value

    def refresh(self, properties: Dict[str, np.ndarray]):
        # TODO: should probably resize based on number of rows but can't rely on that yet.
        pass


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
