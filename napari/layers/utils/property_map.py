from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import validator

from ...utils import Colormap
from ...utils.colormaps import ensure_colormap
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from .color_transformations import ColorType


class ConstantPropertyMap(EventedModel):
    """Maps from a property row to a constant.

    Attributes
    ----------
    constant : Any
        The constant value to always return.
    """

    constant: Any

    def __call__(self, property_row: Dict[str, Any]) -> Any:
        return self.constant


class NamedPropertyMap(EventedModel):
    """Maps from a property row to a property value by name.

    Attributes
    ----------
    name : str
        The name of the property to select from a row.
    """

    name: str

    def __call__(self, property_row: Dict[str, Any]) -> Any:
        return property_row[self.name]


class NamedPropertyDiscreteMap(EventedModel):
    """Maps from a property row to a property value by name to another value defined by a discrete mapping.

    Attributes
    ----------
    name : str
        The name of the property to select from a row.
    discrete_map : dict
        The map from the discrete named property value to the output value.
    """

    name: str
    discrete_map: dict

    def __call__(self, property_row: Dict[str, Any]) -> Any:
        return self.discrete_map.get(property_row[self.name])


class NamedPropertyColorMap(EventedModel):
    """Maps from a property row to a property value by name to another value defined by a discrete mapping.

    Attributes
    ----------
    name : str
        The name of the property to select from a row.
    colormap : Colormap
        The map from the continuous named property value to the output color value.
    """

    name: str
    colormap: Colormap

    def __call__(self, property_row: Dict[str, Any]) -> ColorType:
        return self.colormap.map(property_row[self.name])[0]

    @validator('colormap', pre=True, always=True)
    def _check_colormap(cls, colormap):
        return ensure_colormap(colormap)


class TextFormatPropertyMap(EventedModel):
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


class PropertyTable(EventedModel):
    properties: Dict[str, Array] = {}


class StyleAttribute(EventedModel):
    """Stores a style and its value for each element of a Layer directly.

    Attributes
    ----------
    values : Array
        The style value for each element.
    """

    values: List[Any] = []
    default_value: Any = None
    mapping: Optional[Callable[[Dict[str, Any]], Any]] = None

    def refresh(self, properties: Dict[str, Array]):
        """Updates all values from the given properties.

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        """
        if self.mapping is not None:
            num_values = StyleAttribute._num_values(properties)
            self.values = self._apply(properties, range(0, num_values))

    def add(self, properties: Dict[str, Array], num_to_add: int):
        """Adds a number of a new text values based on the given properties

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        num_to_add : int
            The number of values to add.
        """
        if self.mapping is None:
            self.values.extend([self.default_value] * num_to_add)
        else:
            num_values = StyleAttribute._num_values(properties)
            indices = range(num_values - num_to_add, num_values)
            self.values.extend(self._apply(properties, indices))

    def remove(self, indices: Iterable[int]):
        """Removes some text values by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to remove.
        """
        indices = set(indices)
        self.values = [
            self.values[i] for i in range(len(self.values)) if i not in indices
        ]

    def _apply(self, properties: Dict[str, Array], indices: Iterable[int]):
        return [
            self.mapping(
                {name: column[index] for name, column in properties.items()}
            )
            for index in indices
        ]

    @staticmethod
    def _num_values(properties):
        return (
            len(next(iter(properties.values()))) if len(properties) > 0 else 0
        )
