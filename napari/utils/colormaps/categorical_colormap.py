from typing import Any, Dict, Union

import numpy as np
from pydantic import validator

from ...layers.utils.color_transformations import transform_color_cycle
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from .categorical_colormap_utils import ColorCycle
from .standardize_color import transform_color


def compare_colormap_dicts(cmap_1, cmap_2):

    if len(cmap_1) != len(cmap_2):
        return False
    for k, v in cmap_1.items():
        if k not in cmap_2:
            return False
        if not np.allclose(v, cmap_2[k]):
            return False
    return True


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

    colormap: Dict[Any, Array[np.float32, (4,)]] = {}
    fallback_color: ColorCycle = 'white'

    @validator('colormap', pre=True)
    def _standardize_colormap(cls, v):
        transformed_colormap = {k: transform_color(v)[0] for k, v in v.items()}
        return transformed_colormap

    @validator('fallback_color', pre=True)
    def _standardize_colorcycle(cls, v):
        if isinstance(v, ColorCycle):
            color_cycle = v
        elif isinstance(v, dict):
            color_cycle = ColorCycle(**v)
        else:
            if isinstance(v, str):
                v = [v]
            (
                transformed_color_cycle,
                transformed_colors,
            ) = transform_color_cycle(
                color_cycle=v,
                elem_name='color_cycle',
                default="white",
            )
            color_cycle = ColorCycle(
                values=transformed_colors, cycle=transformed_color_cycle
            )
        return color_cycle

    def map(self, color_properties: Union[list, np.ndarray]) -> np.ndarray:
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
        if isinstance(color_properties, (list, np.ndarray)):
            color_properties = np.asarray(color_properties)
        else:
            color_properties = np.asarray([color_properties])

        # add properties if they are not in the colormap
        color_cycle_keys = [*self.colormap]
        props_in_map = np.in1d(color_properties, color_cycle_keys)
        if not np.all(props_in_map):
            new_prop_values = color_properties[np.logical_not(props_in_map)]
            indices_to_add = np.unique(new_prop_values, return_index=True)[1]
            props_to_add = [
                new_prop_values[index] for index in sorted(indices_to_add)
            ]
            for prop in props_to_add:
                new_color = next(self.fallback_color.cycle)
                self.colormap[prop] = np.squeeze(transform_color(new_color))
        # map the colors
        colors = np.array([self.colormap[x] for x in color_properties])
        return colors

    def __eq__(self, other):
        if isinstance(other, CategoricalColormap):
            if not compare_colormap_dicts(self.colormap, other.colormap):
                return False
            if not np.allclose(
                self.fallback_color.values, other.fallback_color.values
            ):
                return False
            return True
        else:
            return False
