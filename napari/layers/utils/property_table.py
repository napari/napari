from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from ...utils.events import EventedDict, EventedModel
from ...utils.events.custom_types import Array
from .layer_utils import get_current_properties, prepare_properties

if TYPE_CHECKING:
    from pandas import DataFrame


class PropertyColumn(EventedModel):
    """A property defined by a name and values.

    Attributes
    ----------
    name : str
        The name of this property.
    values : np.ndarray
        The actual values of this property.
    choices: np.ndarray
        The possible values of this property.
    default_value: Any
        The value for the next item to be added.
    """

    name: str
    values: Array
    choices: Array
    default_value: Any

    def resize(self, size):
        num_values = len(self.values)
        if size < num_values:
            self.values = np.resize(self.values, size)
        elif size > num_values:
            new_values = np.repeat(
                self.default_value, size - num_values, axis=0
            )
            self.values = np.concatenate((self.values, new_values), axis=0)

    def remove(self, indices):
        self.values = np.delete(self.values, indices, axis=0)

    @property
    def num_values(self):
        return len(self.values)

    @classmethod
    def from_values(cls, name, values):
        values = np.asarray(values)
        choices = np.unique(values)
        default_value = None if len(values) == 0 else values[-1]
        return cls(
            name=name,
            values=values,
            choices=choices,
            default_value=default_value,
        )

    @classmethod
    def from_choices(cls, name, choices):
        choices = np.asarray(choices)
        values = np.empty(0, dtype=choices.dtype)
        default_value = None if len(choices) == 0 else choices[0]
        return cls(
            name=name,
            values=values,
            choices=choices,
            default_value=default_value,
        )


class PropertyTable(EventedDict):
    """Manages a collection of properties."""

    def __init__(self, properties=None):
        super().__init__(data=properties, basetype=PropertyColumn)

    def resize(self, size):
        for prop in self.values():
            prop.resize(size)

    def remove(self, indices):
        for prop in self.values():
            prop.remove(indices)

    @property
    def num_values(self):
        return 0 if len(self) == 0 else next(iter(self.values())).num_values

    @property
    def num_properties(self):
        return len(self)

    @property
    def all_values(self):
        return {prop.name: prop.values for prop in self.values()}

    @property
    def all_choices(self):
        return {prop.name: prop.choices for prop in self.values()}

    @property
    def all_default_values(self):
        return {
            prop.name: np.atleast_1d(prop.default_value)
            for prop in self.values()
        }

    @classmethod
    def from_property_list(cls, property_list):
        return cls(properties={prop.name: prop for prop in property_list})

    @classmethod
    def from_property_arrays(cls, property_arrays):
        properties = {
            name: PropertyColumn.from_values(name, array)
            for name, array in property_arrays.items()
        }
        return cls(properties=properties)

    @classmethod
    def from_property_choices(cls, property_choices):
        properties = {
            name: PropertyColumn.from_choices(name, choices)
            for name, choices in property_choices.items()
        }
        return cls(properties=properties)

    @classmethod
    def from_dataframe(cls, dataframe):
        return cls.from_property_arrays(
            {name: dataframe[name] for name in dataframe}
        )

    @classmethod
    def from_layer_kwargs(
        cls,
        *,
        properties: Optional[Union[Dict[str, Array], 'DataFrame']] = None,
        property_choices: Optional[Dict[str, Array]] = None,
        num_data: Optional[int] = None,
    ):
        properties, property_choices = prepare_properties(
            properties=properties,
            choices=property_choices,
            num_data=num_data,
            save_choices=True,
        )
        current_properties = get_current_properties(
            properties, property_choices
        )
        return cls.from_property_list(
            [
                PropertyColumn(
                    name=name,
                    values=properties[name],
                    choices=property_choices[name],
                    default_value=current_properties[name][0],
                )
                for name in properties
            ]
        )
