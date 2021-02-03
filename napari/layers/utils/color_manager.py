from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from pydantic import root_validator, validator

from ...utils.colormaps import Colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.colormap_utils import ensure_colormap
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ._color_manager_constants import ColorMode
from .color_manager_utils import guess_continuous, is_color_mapped
from .color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color,
    transform_color_with_defaults,
)
from .layer_utils import map_property


@dataclass
class ColorProperties:
    name: str
    values: np.ndarray

    def __eq__(self, other):
        if isinstance(other, ColorProperties):
            names_eq = self.name == other.name
            values_eq = np.array_equal(self.values, other.values)

            eq = names_eq & values_eq
        else:
            eq = False

        return eq


def compare_colormap(cmap_1, cmap_2):
    names_eq = cmap_1.name == cmap_2.name
    colors_eq = np.array_equal(cmap_1.colors, cmap_2.colors)

    return np.all([names_eq, colors_eq])


def compare_categorical_colormap(cmap_1, cmap_2):
    # todo: add real equivalence test
    return False


def compare_color_properties(c_prop_1, c_prop_2):

    names_eq = c_prop_1.name == c_prop_2.name
    values_eq = np.array_equal(c_prop_1.values, c_prop_2.values)

    eq = names_eq & values_eq

    return eq


class ColorManager(EventedModel):
    # fields
    current_color: Optional[np.ndarray] = None
    mode: ColorMode = ColorMode.DIRECT
    color_properties: Optional[ColorProperties] = None
    continuous_colormap: Colormap = 'viridis'
    contrast_limits: Optional[Tuple[float, float]] = None
    categorical_colormap: CategoricalColormap = [0, 0, 0, 1]
    n_colors: Optional[int] = 0
    colors: Array[float, (-1, 4)] = [0, 0, 0, 1]

    __equality_checks__ = {
        'continuous_colormap': compare_colormap,
        'categorical_colormap': compare_categorical_colormap,
        'color_properties': compare_color_properties,
    }

    # validators
    @validator('continuous_colormap', pre=True)
    def _ensure_continuous_colormap(cls, v):
        coerced_colormap = ensure_colormap(v)
        return coerced_colormap

    @validator('categorical_colormap', pre=True)
    def _coerce_categorical_colormap(cls, v):
        if isinstance(v, list) or isinstance(v, np.ndarray):
            fallback_color = v

            # reset the color mapping
            colormap = {}
        elif isinstance(v, dict):
            if ('colormap' in v) or ('fallback_color' in v):
                if 'colormap' in v:
                    colormap = {
                        k: transform_color(v)[0]
                        for k, v in v['colormap'].items()
                    }
                else:
                    colormap = {}
                if 'fallback_color' in v:
                    fallback_color = v['fallback_color']
                else:
                    fallback_color = 'black'
            else:
                colormap = {k: transform_color(v)[0] for k, v in v.items()}
                fallback_color = 'black'
        else:
            raise TypeError('colormap should be an array or dict')

        return CategoricalColormap(
            colormap=colormap, fallback_color=fallback_color
        )

    @validator('color_properties', pre=True)
    def _coerce_color_properties(cls, v):
        if v is None:
            color_properties = v
        elif isinstance(v, dict):
            if len(v) == 0:
                color_properties = None
            elif len(v) == 1:
                name, values = next(iter(v.items()))
                color_properties = ColorProperties(name=name, values=values)
            else:
                raise ValueError(
                    'color_properties should have 0 or 1 key/value pair'
                )
        elif isinstance(v, ColorProperties):
            color_properties = v
        else:
            raise TypeError(
                'color_properties should be a dict or ColorProperties object'
            )

        return color_properties

    @validator('colors', pre=True)
    def _ensure_color_array(cls, v, values):

        return transform_color(v)

    @root_validator(skip_on_failure=True)
    def refresh_colors(cls, values):

        color_mode = values['mode']

        if color_mode == ColorMode.CYCLE:
            color_properties = values['color_properties'].values
            cmap = values['categorical_colormap']
            colors = cmap.map(color_properties)

        elif color_mode == ColorMode.COLORMAP:
            color_properties = values['color_properties'].values
            cmap = values['continuous_colormap']
            if len(color_properties) > 0:
                if values['contrast_limits'] is None:
                    colors, contrast_limits = map_property(
                        prop=color_properties,
                        colormap=cmap,
                    )
                    values['contrast_limits'] = contrast_limits
                else:
                    colors, _ = map_property(
                        prop=color_properties,
                        colormap=cmap,
                        contrast_limits=values['contrast_limits'],
                    )
            else:
                colors = np.empty((0, 4))
                values['color_properties'] = None

            if len(colors) == 0:
                colors = np.empty((0, 4))
        elif color_mode == ColorMode.DIRECT:
            n_colors = values['n_colors']
            if n_colors == 0:
                colors = np.empty((0, 4))
            else:
                transformed_color = transform_color_with_defaults(
                    num_entries=n_colors,
                    colors=values['colors'],
                    elem_name="color",
                    default="white",
                )
                colors = normalize_and_broadcast_colors(
                    n_colors, transformed_color
                )

        # set the current color to the last color/property value
        # if it wasn't already set
        if values['current_color'] is None and len(colors) > 0:
            if color_mode == ColorMode.DIRECT:
                values['current_color'] = colors[-1]
            else:
                property_values = values['color_properties'].values
                values['current_color'] = property_values[-1]
        if values['current_color'] is None and len(colors) == 0:
            if color_mode == ColorMode.DIRECT:
                transformed_color = transform_color_with_defaults(
                    num_entries=n_colors,
                    colors=values['colors'],
                    elem_name="color",
                    default="white",
                )
                values['current_color'] = normalize_and_broadcast_colors(
                    1, transformed_color
                )[0]

        # set the colors
        values['colors'] = colors
        return values

    def add(self, color: Optional[ColorType] = None, n_colors: int = 1):
        """Add colors
        Parameters
        ----------
        color : Optional[ColorType]
            The color to add. If set to None, the value of self.current_color will be used.
            The default value is None.
        n_colors : int
            The number of colors to add. The default value is 1.
        """
        if self.mode == ColorMode.DIRECT:
            if color is None:
                new_color = self.current_color
            else:
                new_color = color
            transformed_color = transform_color_with_defaults(
                num_entries=n_colors,
                colors=new_color,
                elem_name="color",
                default="white",
            )
            broadcasted_colors = normalize_and_broadcast_colors(
                n_colors, transformed_color
            )
            self.colors = np.concatenate((self.colors, broadcasted_colors))
        else:
            # add the new value color_properties
            color_property_name = self.color_properties.name
            new_color_property_values = np.concatenate(
                (self.color_properties.values, np.repeat(color, n_colors)),
                axis=0,
            )
            self.color_properties = ColorProperties(
                name=color_property_name, values=new_color_property_values
            )

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Remove the indicated color elements
        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        selected_indices = list(indices_to_remove)
        if len(selected_indices) > 0:
            if self.mode == ColorMode.DIRECT:
                self.colors = np.delete(self.colors, selected_indices, axis=0)
            else:
                # remove the color_properties
                color_property_name = self.color_properties.name
                new_color_property_values = np.delete(
                    self.color_properties.values, selected_indices
                )
                self.color_properties = ColorProperties(
                    name=color_property_name, values=new_color_property_values
                )


def initialize_color_manager(
    n_colors: int,
    colors: Union[dict, np.ndarray],
    mode,
    continuous_colormap,
    contrast_limits,
    categorical_colormap,
    properties: Dict[str, np.ndarray],
    current_color: Optional[np.ndarray] = None,
    default_color_cycle: np.ndarray = np.array([0, 0, 0, 1]),
) -> ColorManager:
    """Initialize a ColorManager argument from layer kwargs. This is a convenience
    function to coerce possible inputs into ColorManager kwargs

    """
    if isinstance(colors, dict):
        # if the kwargs are passed as a dictionary, unpack them
        color_values = colors.get('colors', None)
        current_color = colors.get('current_color', None)
        mode = colors.get('mode', None)
        color_properties = colors.get('color_properties', None)
        continuous_colormap = colors.get('continuous_colormap', None)
        contrast_limits = colors.get('contrast_limits', None)
        categorical_colormap = colors.get('categorical_colormap', None)
        n_colors = colors.get('n_colors', None)
    else:
        color_values = colors

    if categorical_colormap is None:
        categorical_colormap = deepcopy(default_color_cycle)

    color_kwargs = {
        'categorical_colormap': categorical_colormap,
        'continuous_colormap': continuous_colormap,
        'contrast_limits': contrast_limits,
        'current_color': current_color,
        'n_colors': n_colors,
    }

    if is_color_mapped(color_values, properties):
        color_properties = ColorProperties(
            name=color_values, values=properties[color_values]
        )
        if guess_continuous(color_properties.values):
            mode = ColorMode.COLORMAP
        else:
            mode = ColorMode.CYCLE

        color_kwargs.update(
            {'mode': mode, 'color_properties': color_properties}
        )

    else:
        color_kwargs.update({'mode': ColorMode.DIRECT, 'colors': colors})

    color_manager = ColorManager(**color_kwargs)

    return color_manager
