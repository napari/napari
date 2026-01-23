from typing import Any

import numpy as np
from pydantic import Field, model_validator

from napari.utils.color import ColorValue
from napari.utils.colormaps.categorical_colormap_utils import (
    ColorCycle,
    compare_colormap_dicts,
)
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import EventedModel


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

    @model_validator(mode='before')
    def _validate_args(cls, values):
        if ('colormap' in values) or ('fallback_color' in values):
            if 'colormap' in values:
                colormap = {
                    k: transform_color(v)[0]
                    for k, v in values['colormap'].items()
                }
            else:
                colormap = {}
            fallback_color = values.get('fallback_color', 'white')
        else:
            colormap = {k: transform_color(v)[0] for k, v in values.items()}
            fallback_color = 'white'

        values['colormap'] = colormap
        values['fallback_color'] = fallback_color

        return values

    def __eq__(self, other):
        return (
            isinstance(other, CategoricalColormap)
            and compare_colormap_dicts(self.colormap, other.colormap)
            and np.allclose(
                self.fallback_color.values, other.fallback_color.values
            )
        )
