from typing import Any

import numpy as np

from ...utils.events import EventedModel
from ...utils.translations import trans


class Property(EventedModel):
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
    values: np.ndarray
    choices: np.ndarray
    default_value: Any

    def resize(self, size):
        num_values = len(self.values)
        if size < num_values:
            self.values = np.resize(self.values, size)
        elif size > num_values:
            self.values = np.append(
                self.values, [self.default_value] * (size - num_values)
            )

    def remove(self, indices):
        self.values = np.delete(self.values, indices, axis=0)

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


class PropertyManager:
    """Manages a collection of properties."""

    def __init__(self, properties=None):
        if properties is None:
            properties = {}
        self._properties = properties

    def dict(self):
        return self._properties

    def get(self, name):
        return self._properties[name]

    def add(self, prop):
        self._properties[prop.name] = prop

    def resize(self, size):
        for prop in self._properties.values():
            prop.resize(size)

    def remove(self, indices):
        for prop in self._properties.values():
            prop.remove(indices)

    @property
    def values(self):
        return {prop.name: prop.values for prop in self._properties.values()}

    @property
    def choices(self):
        return {prop.name: prop.choices for prop in self._properties.values()}

    @property
    def default_values(self):
        return {
            prop.name: np.atleast_1d(prop.default_value)
            for prop in self._properties.values()
        }

    @default_values.setter
    def default_values(self, default_values):
        for name, value in default_values.items():
            if name in self._properties:
                self._properties[name].default_value = value

    @classmethod
    def from_property_list(cls, property_list):
        return cls(properties={prop.name: prop for prop in property_list})

    @classmethod
    def from_property_arrays(cls, property_arrays):
        properties = {
            name: Property.from_values(name, array)
            for name, array in property_arrays.items()
        }
        return cls(properties=properties)

    @classmethod
    def from_property_choices(cls, property_choices):
        properties = {
            name: Property.from_choices(name, choices)
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
        cls, *, properties=None, property_choices=None, expected_len=None
    ):
        if properties is not None:
            if isinstance(properties, PropertyManager):
                manager = properties
            elif isinstance(properties, dict):
                manager = cls.from_property_arrays(properties)
            else:
                manager = cls.from_dataframe(properties)
        elif property_choices is not None:
            manager = cls.from_property_choices(property_choices)
        else:
            manager = cls()
        lens = [len(v) for v in manager.values.values()]
        if expected_len is None and len(lens) > 0:
            expected_len = lens[0]
        if any(v != expected_len for v in lens):
            raise ValueError(
                trans._(
                    "the number of items must be equal for all properties",
                    deferred=True,
                )
            )
        return manager
