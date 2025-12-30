from typing import Any

import numpy as np
from pydantic import GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema

from napari._pydantic_compat import Field
from napari.utils.color import ColorValue
from napari.utils.colormaps.categorical_colormap_utils import (
    ColorCycle,
    compare_colormap_dicts,
)
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import EventedModel
from napari.utils.translations import trans


class CategoricalColormap(EventedModel):
    """Colormap that relates categorical values to colors.

    Parameters
    ----------
    colormap : Dict[Any, np.ndarray]
        The mapping between categorical property values and color.
    fallback_color : ColorCycle
        The color to be used in the case that a value is mapped that is not
        in colormap. This can be given as any ColorType and it will be converted
        to a ColorCycle. An array of the values contained in the
        ColorCycle.cycle is stored in ColorCycle.values.
        The default value is a cycle of all white.
    """

    colormap: dict[Any, ColorValue] = Field(default_factory=dict)
    fallback_color: ColorCycle = Field(
        default_factory=lambda: ColorCycle.validate_type('white')
    )

    @model_validator(mode='before')
    @classmethod
    def _preprocess_input(cls, data):
        """Preprocess input data before normal validation."""
        if isinstance(data, cls):
            return data
        if isinstance(data, list | np.ndarray):
            # Array input means fallback_color only
            return {'colormap': {}, 'fallback_color': data}
        if isinstance(data, dict):
            # Check if it's a structured dict with 'colormap' or 'fallback_color' keys
            if ('colormap' in data) or ('fallback_color' in data):
                result = {}
                if 'colormap' in data:
                    result['colormap'] = {
                        k: transform_color(v)[0]
                        for k, v in data['colormap'].items()
                    }
                if 'fallback_color' in data:
                    result['fallback_color'] = data['fallback_color']
                return result
            # Otherwise, treat the entire dict as a colormap
            return {
                'colormap': {k: transform_color(v)[0] for k, v in data.items()},
                'fallback_color': 'white',
            }
        return data

    def map(self, color_properties: list | np.ndarray) -> np.ndarray:
        """Map an array of values to an array of colors
        Parameters
        ----------
        color_properties : Union[list, np.ndarray]
            The property values to be converted to colors.
        Returns
        -------
        colors : np.ndarray
            An Nx4 color array where N is the number of property values provided.
        """
        if isinstance(color_properties, list | np.ndarray):
            color_properties = np.asarray(color_properties)
        else:
            color_properties = np.asarray([color_properties])

        # add properties if they are not in the colormap
        color_cycle_keys = [*self.colormap]
        props_in_map = np.isin(color_properties, color_cycle_keys)
        if not np.all(props_in_map):
            new_prop_values = color_properties[np.logical_not(props_in_map)]
            indices_to_add = np.unique(new_prop_values, return_index=True)[1]
            props_to_add = [
                new_prop_values[index]
                for index in sorted(int(x) for x in indices_to_add)
            ]
            for prop in props_to_add:
                new_color = next(self.fallback_color.cycle)
                self.colormap[prop] = ColorValue(new_color)
        # map the colors
        colors = np.array([self.colormap[x] for x in color_properties])
        return colors

    @classmethod
    def from_array(cls, fallback_color):
        """Create from array."""
        return cls(colormap={}, fallback_color=fallback_color)

    @classmethod
    def from_dict(cls, params: dict):
        """Create from dict."""
        return cls(**params) if ('colormap' in params or 'fallback_color' in params) else cls(colormap=params)

    @classmethod
    def validate_type(cls, val):
        """Legacy validation method kept for compatibility."""
        if isinstance(val, cls):
            return val
        if isinstance(val, list | np.ndarray):
            return cls(colormap={}, fallback_color=val)
        if isinstance(val, dict):
            if ('colormap' in val) or ('fallback_color' in val):
                return cls(**val)
            return cls(colormap=val)
        raise TypeError(
            trans._(
                'colormap should be an array or dict',
                deferred=True,
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, CategoricalColormap)
            and compare_colormap_dicts(self.colormap, other.colormap)
            and np.allclose(
                self.fallback_color.values, other.fallback_color.values
            )
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Define how Pydantic V2 should validate this type when used as a field."""
        # Use a before validator to preprocess input, then use the normal model schema
        return core_schema.no_info_before_validator_function(
            cls._preprocess_for_field,
            handler(source_type),
        )

    @classmethod
    def _preprocess_for_field(cls, val):
        """Preprocess input when CategoricalColormap is used as a field type."""
        if isinstance(val, cls):
            return val
        if isinstance(val, list | np.ndarray):
            # Array input means fallback_color only
            return {'colormap': {}, 'fallback_color': val}
        if isinstance(val, dict):
            # Check if it's a structured dict with 'colormap' or 'fallback_color' keys
            if ('colormap' in val) or ('fallback_color' in val):
                result = {}
                if 'colormap' in val:
                    result['colormap'] = {
                        k: transform_color(v)[0]
                        for k, v in val['colormap'].items()
                    }
                if 'fallback_color' in val:
                    result['fallback_color'] = val['fallback_color']
                return result
            # Otherwise, treat the entire dict as a colormap
            return {
                'colormap': {k: transform_color(v)[0] for k, v in val.items()},
                'fallback_color': 'white',
            }
        return val
