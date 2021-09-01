from typing import Any, Dict, Generic, Iterable, List, TypeVar

from pydantic import validator

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from .color_transformations import ColorType

OutputType = TypeVar('OutputType')


class PropertyMap(Generic[OutputType], EventedModel):
    values: List[OutputType] = []

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        pass

    def refresh(self, properties: Dict[str, Array]):
        """Updates all values from the given properties.

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        """
        num_values = _num_rows(properties)
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
        num_values = _num_rows(properties)
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
            self({name: column[index] for name, column in properties.items()})
            for index in indices
        ]

    @staticmethod
    def _num_values(properties):
        return (
            len(next(iter(properties.values()))) if len(properties) > 0 else 0
        )

    @classmethod
    def from_format_string(cls, format_string: str):
        return TextFormatPropertyMap(format_string=format_string)

    @classmethod
    def from_property(cls, property_name: OutputType):
        return NamedPropertyMap(name=property_name)

    @classmethod
    def from_constant(cls, constant: OutputType):
        return ConstantPropertyMap(constant=constant)

    @classmethod
    def from_discrete_map(
        cls, property_name: str, discrete_map: Dict[Any, OutputType]
    ):
        return NamedPropertyDiscreteMap(
            name=property_name, discrete_map=discrete_map
        )

    @classmethod
    def from_colormap(cls, property_name: str, colormap: ValidColormapArg):
        return NamedPropertyColorMap(name=property_name, colormap=colormap)

    @classmethod
    def from_iterable(
        cls, iterable: Iterable[OutputType], default_value: OutputType
    ):
        return DirectPropertyMap(
            values=list(iterable), default_value=default_value
        )


def _num_rows(properties: Dict[str, Array]) -> int:
    return len(next(iter(properties.values()))) if len(properties) > 0 else 0


class DirectPropertyMap(PropertyMap[OutputType], EventedModel):
    default_value: OutputType

    def refresh(self, properties: Dict[str, Array]):
        # May want to resize values based on size of properties.
        pass

    def add(self, properties: Dict[str, Array], num_to_add: int):
        self.values.extend([self.default_value] * num_to_add)


class ConstantPropertyMap(PropertyMap[OutputType], EventedModel):
    """Maps from a property row to a constant.

    Attributes
    ----------
    constant : Any
        The constant value to always return.
    """

    constant: OutputType

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return self.constant


class NamedPropertyMap(PropertyMap[OutputType], EventedModel):
    """Maps from a property row to a property value by name.

    Attributes
    ----------
    name : str
        The name of the property to select from a row.
    """

    name: str

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return property_row[self.name]


class NamedPropertyDiscreteMap(PropertyMap[OutputType], EventedModel):
    """Maps from a property row to a property value by name to another value defined by a discrete mapping.

    Attributes
    ----------
    name : str
        The name of the property to select from a row.
    discrete_map : dict
        The map from the discrete named property value to the output value.
    """

    name: str
    discrete_map: dict[Any, OutputType]

    def __call__(self, property_row: Dict[str, Any]) -> OutputType:
        return self.discrete_map.get(property_row[self.name])


class NamedPropertyColorMap(PropertyMap[ColorType], EventedModel):
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
    def _check_colormap(cls, colormap: ValidColormapArg) -> Colormap:
        return ensure_colormap(colormap)


class TextFormatPropertyMap(PropertyMap[str], EventedModel):
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
