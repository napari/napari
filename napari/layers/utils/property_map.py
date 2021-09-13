from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterable, List, TypeVar, Union

import numpy as np
from pydantic import validator
from pydantic.generics import GenericModel

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.events import EventedModel
from .color_transformations import ColorType

OutputType = TypeVar('OutputType')


class PropertyMap(EventedModel, GenericModel, Generic[OutputType], ABC):
    values: List[OutputType] = []

    @abstractmethod
    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        pass

    def refresh(self, properties: Dict[str, np.ndarray]):
        """Updates all values from the given properties.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.
        """
        num_values = _num_rows(properties)
        self.values = self._apply(properties, range(0, num_values))

    def add(self, properties: Dict[str, np.ndarray], num_to_add: int):
        """Adds a number of a new values based on the given properties.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.
        num_to_add : int
            The number of values to add.
        """
        num_values = _num_rows(properties)
        indices = range(num_values - num_to_add, num_values)
        self.values.extend(self._apply(properties, indices))

    def remove(self, indices: Iterable[int]):
        """Removes some values by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to remove.
        """
        indices = set(indices)
        self.values = [
            self.values[i] for i in range(len(self.values)) if i not in indices
        ]

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ):
        return [
            self({name: column[index] for name, column in properties.items()})
            for index in indices
        ]


# TODO: if there are no property columns, this will return 0
# even if there are some data.
def _num_rows(properties: Dict[str, np.ndarray]) -> int:
    return len(next(iter(properties.values()))) if len(properties) > 0 else 0


class DirectPropertyMap(PropertyMap[OutputType], Generic[OutputType]):
    default_value: OutputType

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return self.default_value

    def refresh(self, properties: Dict[str, np.ndarray]):
        pass
        # TODO: should probably resize based on number of rows.
        # num_rows = _num_rows(properties)
        # num_values = len(self.values)
        # if num_values > num_rows:
        #    self.remove(range(num_rows, num_values))
        # elif num_values < num_rows:
        #    self.add(num_rows - num_values)


class ConstantPropertyMap(PropertyMap[OutputType], Generic[OutputType]):
    """Maps from a property row to a constant.

    Attributes
    ----------
    constant : Any
        The constant value to always return.
    """

    constant: OutputType

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return self.constant


class NamedPropertyMap(PropertyMap[OutputType], Generic[OutputType]):
    """Maps from a property row to a property value by name.

    Attributes
    ----------
    property_name : str
        The name of the property to select from a row.
    """

    property_name: str

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return property_row[self.property_name]


class NamedPropertyDiscreteMap(PropertyMap[OutputType], Generic[OutputType]):
    """Maps from a property row to a property value by name to another value defined by a discrete mapping.

    Attributes
    ----------
    property_name : str
        The name of the property to select from a row.
    discrete_map : dict
        The map from the discrete named property value to the output value.
    """

    property_name: str
    discrete_map: dict[Any, OutputType]

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return self.discrete_map.get(property_row[self.property_name])


class NamedPropertyColorMap(PropertyMap[ColorType]):
    """Maps from a property row to a property value by name to another value defined by a discrete mapping.

    Attributes
    ----------
    name : str
        The name of the property to select from a row.
    colormap : Colormap
        The map from the continuous named property value to the output color value.
    """

    property_name: str
    colormap: Colormap

    def __call__(self, property_row: Dict[str, Any]) -> ColorType:
        return self.colormap.map(property_row[self.property_name])[0]

    @validator('colormap', pre=True, always=True)
    def _check_colormap(cls, colormap: ValidColormapArg) -> Colormap:
        return ensure_colormap(colormap)


class TextFormatPropertyMap(PropertyMap[str]):
    """Maps from a property row to a formatted string containing property names.

    Attributes
    ----------
    format_string : str
        The format string as described in str.format. The format placeholders
        should only reference property names.
    """

    format_string: str

    def __call__(self, property_row: Dict[str, Any]) -> str:
        return self.format_string.format(**property_row)


ColorPropertyMap = Union[
    NamedPropertyColorMap,
    NamedPropertyDiscreteMap[ColorType],
    NamedPropertyMap[ColorType],
    DirectPropertyMap[ColorType],
    ConstantPropertyMap[ColorType],
]


StringPropertyMap = Union[
    TextFormatPropertyMap,
    NamedPropertyDiscreteMap[str],
    NamedPropertyMap[str],
    DirectPropertyMap[str],
    ConstantPropertyMap[str],
]
