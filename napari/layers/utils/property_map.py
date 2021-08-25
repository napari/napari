from typing import Any, Callable, Dict, List

from ...utils.events import EventedModel


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


class PropertyMapStore(EventedModel):
    """Stores a property row map, as well the values generated from that map.

    Attributes
    ----------
    mapping : Callable[[Dict[str, Any]], Any]
        A mapping from a property row to any value.
    values : Array
        The values generated from the mapping.
    """

    mapping: Callable[[Dict[str, Any]], Any]
    values: List[Any] = []

    def refresh(self, properties):
        """Updates all values from the given properties.

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        """
        num_values = PropertyMapStore._num_values(properties)
        self.values = self._apply(properties, range(0, num_values))

    def add(self, properties, num_to_add):
        """Adds a number of a new text values based on the given properties

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        num_to_add : int
            The number of values to add.
        """
        num_values = PropertyMapStore._num_values(properties)
        indices = range(num_values - num_to_add, num_values)
        self.values.extend(self._apply(properties, indices))

    def remove(self, indices):
        """Removes some text values by index.

        Parameters
        ----------
        indices : Sequence[int]
            The indices to remove.
        """
        indices = set(indices)
        self.values = [
            self.values[i] for i in range(len(self.values)) if i not in indices
        ]

    def _apply(self, properties, indices):
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
