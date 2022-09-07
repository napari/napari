from typing import Any, Dict, Union

import numpy as np

from napari.utils.color import ColorValue

from ...utils.events import EventedModel
from ..translations import trans
from .categorical_colormap_utils import ColorCycle, compare_colormap_dicts
from .standardize_color import transform_color


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

    colormap: Dict[Any, ColorValue] = {}
    fallback_color: ColorCycle = ColorCycle.validate_type('white')

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

    @classmethod
    def from_array(cls, fallback_color):
        return cls(fallback_color=fallback_color)

    @classmethod
    def from_dict(cls, params: dict):
        if ('colormap' in params) or ('fallback_color' in params):
            if 'colormap' in params:
                colormap = {
                    k: transform_color(v)[0]
                    for k, v in params['colormap'].items()
                }
            else:
                colormap = {}
            if 'fallback_color' in params:
                fallback_color = params['fallback_color']
            else:
                fallback_color = 'white'
        else:
            colormap = {k: transform_color(v)[0] for k, v in params.items()}
            fallback_color = 'white'

        return cls(colormap=colormap, fallback_color=fallback_color)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, cls):
            return val
        if isinstance(val, list) or isinstance(val, np.ndarray):
            return cls.from_array(val)
        elif isinstance(val, dict):
            return cls.from_dict(val)
        else:
            raise TypeError(
                trans._(
                    'colormap should be an array or dict',
                    deferred=True,
                )
            )

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
