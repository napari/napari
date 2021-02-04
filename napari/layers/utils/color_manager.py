from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pydantic import root_validator, validator

from ...utils.colormaps import Colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.colormap_utils import ensure_colormap
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ._color_manager_constants import ColorMode
from .color_manager_utils import (
    guess_continuous,
    is_color_mapped,
    map_property,
)
from .color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color,
    transform_color_with_defaults,
)


@dataclass
class ColorProperties:
    name: str
    values: np.ndarray
    current_value: Optional[Any] = None

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
    if np.array_equal(
        cmap_1.fallback_color.values, cmap_2.fallback_color.values
    ):
        return True
    else:
        return False


def compare_color_properties(c_prop_1, c_prop_2):
    if (c_prop_1 is None) and (c_prop_2 is None):
        return True
    elif (c_prop_1 is None) != (c_prop_2 is None):
        return False
    else:
        names_eq = c_prop_1.name == c_prop_2.name
        values_eq = np.array_equal(c_prop_1.values, c_prop_2.values)

        eq = names_eq & values_eq

    return eq


def compare_colors(color_1, color_2):
    return np.allclose(color_1, color_2)


def compare_contrast_limits(clim_1, clim_2):
    if (clim_1 is None) and (clim_2 is None):
        return True
    elif (clim_1 is None) != (clim_2 is None):
        return False
    else:
        return np.allclose(clim_1, clim_2)


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
        'contrast_limits': compare_contrast_limits,
        'current_color': compare_colors,
        'colors': compare_colors,
    }

    # validators
    @validator('continuous_colormap', pre=True)
    def _ensure_continuous_colormap(cls, v):
        coerced_colormap = ensure_colormap(v)
        return coerced_colormap

    @validator('categorical_colormap', pre=True)
    def _coerce_categorical_colormap(cls, v):
        if isinstance(v, CategoricalColormap):
            return v
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
            else:
                try:
                    color_properties = ColorProperties(**v)
                except ValueError:
                    print(
                        'color_properties dictionary should have keys: name, value, and optionally current_value'
                    )
                    raise

        elif isinstance(v, ColorProperties):
            color_properties = v
        else:
            raise TypeError(
                'color_properties should be None, a dict, or ColorProperties object'
            )

        return color_properties

    @validator('colors', pre=True)
    def _ensure_color_array(cls, v, values):
        if len(v) > 0:
            return transform_color(v)
        else:
            return np.empty((0, 4))

    @root_validator(skip_on_failure=True)
    def refresh_colors(cls, values):

        color_mode = values['mode']

        if color_mode == ColorMode.CYCLE:
            color_properties = values['color_properties'].values
            cmap = values['categorical_colormap']
            if len(color_properties) == 0:
                colors = np.empty((0, 4))
                current_prop_value = values['color_properties'].current_value
                if current_prop_value is not None:
                    values['current_color'] = cmap.map(current_prop_value)[0]
            else:
                colors = cmap.map(color_properties)
            values['categorical_colormap'] = cmap

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
                current_prop_value = values['color_properties'].current_value
                if current_prop_value is not None:
                    values['current_color'] = cmap.map(current_prop_value)[0]

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
            values['current_color'] = colors[-1]
            if color_mode in [ColorMode.CYCLE, ColorMode.COLORMAP]:
                property_values = values['color_properties']
                property_values.current_value = property_values.values[-1]
                values['color_properties'] = property_values
        if values['current_color'] is None and len(colors) == 0:
            if color_mode == ColorMode.DIRECT:
                transformed_color = transform_color_with_defaults(
                    num_entries=values['n_colors'],
                    colors=values['colors'],
                    elem_name="color",
                    default="white",
                )
                values['current_color'] = normalize_and_broadcast_colors(
                    1, transformed_color
                )[0]

        # set the colors
        values['n_colors'] = len(colors)
        values['colors'] = colors
        return values

    def add(
        self,
        color: Optional[ColorType] = None,
        n_colors: int = 1,
        update_clims: bool = False,
    ):
        """Add colors
        Parameters
        ----------
        color : Optional[ColorType]
            The color to add. If set to None, the value of self.current_color will be used.
            The default value is None.
        n_colors : int
            The number of colors to add. The default value is 1.
        update_clims : bool
            If in colormap mode, update the contrast limits when adding the new values
            (i.e., reset the range to 0-new_max_value).
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
            current_value = self.color_properties.current_value
            if color is None:
                color = current_value
            new_color_property_values = np.concatenate(
                (self.color_properties.values, np.repeat(color, n_colors)),
                axis=0,
            )
            self.color_properties = ColorProperties(
                name=color_property_name,
                values=new_color_property_values,
                current_value=current_value,
            )

            if update_clims and self.mode == ColorMode.COLORMAP:
                self.contrast_limits = None

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

    def __eq__(self, other):
        current_color = self.__equality_checks__['current_color'](
            self.current_color, other.current_color
        )
        mode = self.mode == other.mode
        color_properties = self.__equality_checks__['color_properties'](
            self.color_properties, other.color_properties
        )
        continuous_colormap = self.__equality_checks__['continuous_colormap'](
            self.continuous_colormap, other.continuous_colormap
        )
        contrast_limits = self.__equality_checks__['contrast_limits'](
            self.contrast_limits, other.contrast_limits
        )
        categorical_colormap = self.__equality_checks__[
            'categorical_colormap'
        ](self.categorical_colormap, other.categorical_colormap)
        n_colors = self.n_colors == other.n_colors
        colors = self.__equality_checks__['colors'](self.colors, other.colors)

        return np.all(
            [
                current_color,
                mode,
                color_properties,
                continuous_colormap,
                contrast_limits,
                categorical_colormap,
                n_colors,
                colors,
            ]
        )


def initialize_color_manager(
    colors: Union[dict, str, np.ndarray],
    properties: Dict[str, np.ndarray],
    n_colors: Optional[int] = None,
    continuous_colormap: Optional[Union[str, Colormap]] = None,
    contrast_limits: Optional[Tuple[float, float]] = None,
    categorical_colormap: Optional[
        Union[CategoricalColormap, list, np.ndarray]
    ] = None,
    mode: Optional[Union[ColorMode, str]] = None,
    current_color: Optional[np.ndarray] = None,
    default_color_cycle: np.ndarray = np.array([0, 0, 0, 1]),
) -> ColorManager:
    """Initialize a ColorManager object from layer kwargs. This is a convenience
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

        if isinstance(color_properties, str):
            # if the color properties were given as a property name,
            # coerce into ColorProperties
            try:
                prop_values = properties[color_properties]
                prop_name = color_properties
                color_properties = ColorProperties(
                    name=prop_name, values=prop_values
                )
            except KeyError:
                print(
                    'if color_properties is a string, it should be a property name'
                )
                raise
    else:
        color_values = colors
        color_properties = None

    if categorical_colormap is None:
        categorical_colormap = deepcopy(default_color_cycle)

    color_kwargs = {
        'categorical_colormap': categorical_colormap,
        'continuous_colormap': continuous_colormap,
        'contrast_limits': contrast_limits,
        'current_color': current_color,
        'n_colors': n_colors,
    }

    if color_properties is None:
        if is_color_mapped(color_values, properties):
            if n_colors == 0:
                color_properties = ColorProperties(
                    name=color_values,
                    values=np.empty(0),
                    current_value=properties[color_values][0],
                )
            else:
                color_properties = ColorProperties(
                    name=color_values, values=properties[color_values]
                )
            if mode is None:
                if guess_continuous(color_properties.values):
                    mode = ColorMode.COLORMAP
                else:
                    mode = ColorMode.CYCLE

            color_kwargs.update(
                {'mode': mode, 'color_properties': color_properties}
            )

        else:
            if len(color_values) == 0:
                color_kwargs.update({'mode': ColorMode.DIRECT})
            else:
                color_kwargs.update(
                    {'mode': ColorMode.DIRECT, 'colors': color_values}
                )
    else:
        color_kwargs.update(
            {'mode': mode, 'color_properties': color_properties}
        )

    color_manager = ColorManager(**color_kwargs)

    return color_manager
