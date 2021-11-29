from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from napari.layers.utils.layer_utils import coerce_current_properties
from napari.utils.events.custom_types import Array


class PropertyTable:
    def __init__(self, data=None):
        self.data = pd.DataFrame(data)
        self.default_values = {
            name: _get_default_value_from_series(series)
            for name, series in self.data.items()
        }

    def resize(self, size: int):
        if size < self.num_values:
            self.remove(range(size, self.num_values))
        elif size > self.num_values:
            num_append = size - self.num_values
            to_append = pd.DataFrame(
                {
                    name: np.repeat(
                        self._default_values[name], num_append, axis=0
                    )
                    for name in self.data
                },
                index=range(num_append),
            )
            self.append(to_append)

    def append(self, data: pd.DataFrame):
        self.data = self.data.append(data, ignore_index=True)

    def remove(self, indices):
        self.data = self.data.drop(labels=indices, axis=0).reset_index(
            drop=True
        )

    @property
    def num_values(self):
        return self.data.shape[0]

    @property
    def num_properties(self):
        return self.data.shape[1]

    @property
    def default_values(self) -> Dict[str, np.ndarray]:
        return self._default_values

    @default_values.setter
    def default_values(self, default_values):
        # TODO: coerce and check consistency with data.
        self._default_values = coerce_current_properties(default_values)

    @property
    def choices(self) -> Dict[str, np.ndarray]:
        # TODO: should we copy categories?
        return {
            name: series.dtype.categories
            for name, series in self.data.items()
            if isinstance(series.dtype, pd.CategoricalDtype)
        }

    @property
    def values(self) -> Dict[str, np.ndarray]:
        # TODO: Should we always pass copy=True to ensure the return value does
        # not have an effect when modified in-place?
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
        # Provide an explicit index when num_data is provided to error check the properties data length.
        index = None if num_data is None else range(num_data)
        data = pd.DataFrame(data=properties, index=index)
        return cls(data)


def _get_default_value_from_series(series):
    if series.size > 0:
        return series.iloc[-1]
    if isinstance(series.dtype, pd.CategoricalDtype):
        choices = series.dtype.categories
        if choices.size > 0:
            return choices[0]
    return None
