from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterable, List, TypeVar, Union

import numpy as np
from pydantic import validator
from pydantic.generics import GenericModel

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from .color_transformations import ColorType

OutputType = TypeVar('OutputType')


class StyleEncoding(EventedModel, GenericModel, Generic[OutputType], ABC):
    values: Array = []

    @abstractmethod
    def apply_to_row(self, property_row: Dict[str, Any]) -> OutputType:
        pass

    def _coerce_output(self, output: OutputType) -> OutputType:
        return output

    def refresh(self, properties: Dict[str, np.ndarray]):
        """Updates all values from the given properties.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.
        """
        num_values = _num_rows(properties)
        self.values = np.array(
            self._apply_to_table(properties, range(0, num_values))
        )

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
        to_add = self._apply_to_table(properties, indices)
        if len(self.values) == 0:
            self.values = to_add
        else:
            self.values = np.concatenate((self.values, to_add), axis=0)

    def remove(self, indices: Iterable[int]):
        """Removes some values by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to remove.
        """
        self.values = np.delete(self.values, list(indices), axis=0)

    def _apply_to_table(
        self, properties: Dict[str, np.ndarray], indices: Iterable[int]
    ) -> List[OutputType]:
        return [
            self._coerce_output(
                self.apply_to_row(
                    {
                        name: column[index]
                        for name, column in properties.items()
                    }
                )
            )
            for index in indices
        ]


# TODO: if there are no property columns, this will return 0
# even if there are some data.
def _num_rows(properties: Dict[str, np.ndarray]) -> int:
    return len(next(iter(properties.values()))) if len(properties) > 0 else 0


class DirectEncoding(StyleEncoding[OutputType], Generic[OutputType]):
    default_value: OutputType

    def apply_to_row(self, property_row: Dict[str, Any]) -> OutputType:
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


class ConstantEncoding(StyleEncoding[OutputType], Generic[OutputType]):
    """Maps from a property row to a constant.

    Attributes
    ----------
    constant : OutputType
        The constant value to always return.
    """

    constant: OutputType

    def apply_to_row(self, property_row: Dict[str, Any]) -> OutputType:
        return self.constant


class IdentityEncoding(StyleEncoding[OutputType], Generic[OutputType]):
    """Maps from a property row to a property value by name.

    Attributes
    ----------
    property_name : str
        The name of the property to select from a row.
    """

    property_name: str

    def apply_to_row(self, property_row: Dict[str, Any]) -> OutputType:
        return property_row[self.property_name]


class DiscreteEncoding(StyleEncoding[OutputType], Generic[OutputType]):
    """Maps from a property row to a property value by name to another value defined by a discrete mapping.

    Attributes
    ----------
    property_name : str
        The name of the property to select from a row.
    mapping : dict
        The map from the discrete named property value to the output value.
    """

    property_name: str
    mapping: dict[Any, OutputType]

    def apply_to_row(self, property_row: Dict[str, Any]) -> OutputType:
        return self.mapping.get(property_row[self.property_name])


class DirectColorEncoding(DirectEncoding[ColorType]):
    def _coerce_output(self, output):
        return transform_color(output)[0]

    @validator('values', pre=True, always=True)
    def _check_values(cls, values):
        return (
            np.empty((0, 4)) if len(values) == 0 else transform_color(values)
        )


class IdentityColorEncoding(IdentityEncoding[ColorType]):
    def _coerce_output(self, output):
        return transform_color(output)[0]


class ConstantColorEncoding(ConstantEncoding[ColorType]):
    def _coerce_output(self, output):
        return transform_color(output)[0]


class DiscreteColorEncoding(StyleEncoding[ColorType]):
    property_name: str
    mapping: CategoricalColormap

    def apply_to_row(self, property_row: Dict[str, Any]) -> ColorType:
        return self.mapping.map(property_row[self.property_name])[0]


class ContinuousColorEncoding(StyleEncoding[ColorType]):
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


ConstantColorEncoding.__eq_operators__['constant'] = np.array_equal
DirectColorEncoding.__eq_operators__['default_value'] = np.array_equal


class FormatStringEncoding(StyleEncoding[str]):
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


class DirectStringEncoding(DirectEncoding[ColorType]):
    def _coerce_output(self, output):
        return str(output)


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
