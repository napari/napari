from typing import Any, Callable, Dict, List

import numpy as np

from ...utils.events import EventedModel


class ConstantPropertyMap(EventedModel):
    constant: Any

    def __call__(self, property_row: Dict[str, Any]) -> Any:
        return self.constant


class NamedPropertyMap(EventedModel):
    name: str

    def __call__(self, property_row: Dict[str, Any]) -> Any:
        return property_row[self.name]


class TextFormatPropertyMap(EventedModel):
    format_string: str

    def __call__(self, property_row: Dict[str, Any]) -> str:
        return self.format_string.format(**property_row)


class PropertyMapStore(EventedModel):
    mapping: Callable[[Dict[str, Any]], Any]
    values: List[Any] = []

    def refresh(self, properties, num_values=None):
        """Updates all or some text values from the given properties."""
        indices = range(
            0, num_values or PropertyMapStore._num_values(properties)
        )
        self.values = self.apply(properties, indices)

    def add(self, properties, num_add):
        """Adds a number of a new text values based on the given properties."""
        num_values = PropertyMapStore._num_values(properties)
        indices = range(num_values - num_add, num_values)
        self.values.extend(self.apply(properties, indices))

    def remove(self, indices):
        # TODO: fix this properly.
        if np.max(list(indices)) >= len(self.values):
            return
        indices = set(indices)
        self.values = [
            self.values[i] for i in range(len(self.values)) if i not in indices
        ]

    @staticmethod
    def _num_values(properties):
        return (
            len(next(iter(properties.values()))) if len(properties) > 0 else 0
        )

    def apply(self, properties, indices):
        if indices is None:
            indices = range(0, PropertyMapStore._num_values(properties))
        return [
            self.mapping(
                {name: column[index] for name, column in properties.items()}
            )
            for index in indices
        ]
