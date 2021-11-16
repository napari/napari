from typing import Dict, Optional, Union

import pandas as pd

from napari.utils.events.custom_types import Array


class PropertyTable:
    """Manages a collection of properties."""

    def __init__(self, data=None):
        self.data = pd.DataFrame(data)
        self._default_values = {
            name: PropertyTable._get_default_value_from_series(series)
            for name, series in self.data.items()
        }

    @staticmethod
    def _get_default_value_from_series(series):
        if series.size > 0:
            return series.iloc[-1]
        if isinstance(series.dtype, pd.CategoricalDtype):
            choices = series.dtype.categories
            if choices.size > 0:
                return choices[0]
        return None

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
        if property_choices is not None:
            properties = pd.DataFrame(data=properties)
            for name, choices in property_choices.items():
                dtype = pd.CategoricalDtype(categories=choices)
                num_values = (
                    properties.shape[0] if num_data is None else num_data
                )
                values = (
                    properties[name]
                    if name in properties
                    else [None] * num_values
                )
                properties[name] = pd.Series(values, dtype=dtype)
        # Provide an explicit when num_data is provided to error check the properties data length.
        index = None if num_data is None else range(num_data)
        data = pd.DataFrame(data=properties, index=index)
        return cls(data)


def _infer_num_data(
    properties: Optional[Union[Dict[str, Array], pd.DataFrame]],
    num_data: Optional[int],
) -> int:
    if num_data is not None:
        return num_data
    if len(properties) > 0:
        return len(next(iter(properties)))
    return 0
