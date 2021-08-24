from typing import Dict, Optional, Union

import pandas as pd

from napari.utils.events.custom_types import Array


class PropertyTable:
    """Manages a collection of properties."""

    def __init__(self, data=None, default_values=None):
        self.data = pd.DataFrame(data)
        if default_values is None:
            self._default_values = {
                name: self.data[name].iloc[-1] if self.num_values > 0 else None
                for name in self.data
            }
        else:
            self._default_values = default_values

    def resize(self, size):
        if size < self.num_values:
            self.remove(range(size, self.num_values))
        elif size > self.num_values:
            to_append = pd.DataFrame(
                {
                    name: [self._default_values[name]]
                    * (size - self.num_values)
                    for name in self.data
                }
            )
            self.data = self.data.append(to_append, ignore_index=True)

    def remove(self, indices):
        self.data = self.data.drop(labels=indices, axis=0)

    @property
    def num_values(self):
        return self.data.shape[0]

    @property
    def num_properties(self):
        return self.data.shape[1]

    @property
    def default_values(self):
        return self._default_values

    @property
    def choices(self):
        return {
            name: series.dtype.categories
            for name, series in self.data.items()
            if isinstance(series.dtype, pd.CategoricalDtype)
        }

    @property
    def values(self):
        return {name: series.to_numpy() for name, series in self.data.items()}

    @classmethod
    def from_layer_kwargs(
        cls,
        *,
        properties: Optional[Union[Dict[str, Array], pd.DataFrame]] = None,
        property_choices: Optional[Dict[str, Array]] = None,
        num_data: Optional[int] = None,
    ):
        # TODO: do something with property_choices even if it just means issuing a warning.
        index = None if num_data is None else range(num_data)
        data = pd.DataFrame(data=properties, index=index)
        return cls(data)
