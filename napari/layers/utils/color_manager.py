import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import numpy as np
from pydantic import root_validator, validator

from ...utils.colormaps import Colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.colormap_utils import ensure_colormap
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ._color_manager_constants import ColorMode
from .color_manager_utils import (
    _validate_colormap_mode,
    _validate_cycle_mode,
    guess_continuous,
    is_color_mapped,
)
from .color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color,
    transform_color_with_defaults,
)
from .property_table import PropertyColumn, PropertyTable


class ColorManager(EventedModel):
    """A class for controlling the display colors for annotations in napari.

    Attributes
    ----------
    current_color : Optional[np.ndarray]
        A (4,) color array for the color of the next items to be added.
    mode : ColorMode
        The mode for setting colors.

        ColorMode.DIRECT: colors are set by passing color values to ColorManager.colors
        ColorMode.COLORMAP: colors are set via the continuous_colormap applied to the
                            color_properties
        ColorMode.CYCLE: colors are set vie the categorical_colormap appied to the
                         color_properties. This should be used for categorical
                         properties only.
     color_properties : Optional[PropertyColumn]
        The property values that are used for setting colors in ColorMode.COLORMAP
        and ColorMode.CYCLE. The Property dataclass has 3 fields: name,
        values, and current_value. name (str) is the name of the property being used.
        values (np.ndarray) is an array containing the property values.
        current_value contains the value for the next item to be added. color_properties
        can be set as either a ColorProperties object or a dictionary where the keys are
        the field values and the values are the field values (i.e., a dictionary that would
        be valid in ColorProperties(**input_dictionary) ).
    continuous_colormap : Colormap
        The napari colormap object used in ColorMode.COLORMAP mode. This can also be set
        using the name of a known colormap as a string.
    contrast_limits : Tuple[float, float]
        The min and max value for the colormap being applied to the color_properties
        in ColorMonde.COLORMAP mode. Set as a tuple (min, max).
    categorical_colormap : CategoricalColormap
        The napari CategoricalColormap object used in ColorMode.CYCLE mode.
        To set a direct mapping between color_property values and colors,
        pass a dictionary where the keys are the property values and the
        values are colors (either string names or (4,) color arrays).
        To use a color cycle, pass a list or array of colors. You can also
        pass the CategoricalColormap keyword arguments as a dictionary.
    colors : np.ndarray
        The colors in a Nx4 color array, where N is the number of colors.
    """

    current_color: Optional[Array[float, (4,)]] = None
    color_mode: ColorMode = ColorMode.DIRECT
    color_properties: Optional[PropertyColumn] = None
    continuous_colormap: Colormap = 'viridis'
    contrast_limits: Optional[Tuple[float, float]] = None
    categorical_colormap: CategoricalColormap = [0, 0, 0, 1]
    colors: Array[float, (-1, 4)] = []

    @validator('continuous_colormap', pre=True)
    def _ensure_continuous_colormap(cls, v):
        coerced_colormap = ensure_colormap(v)
        return coerced_colormap

    @validator('colors', pre=True)
    def _ensure_color_array(cls, v, values):
        if len(v) > 0:
            return transform_color(v)
        else:
            return np.empty((0, 4))

    @validator('current_color', pre=True)
    def _coerce_current_color(cls, v):
        if v is None:
            return v
        elif len(v) == 0:
            return None
        else:
            return transform_color(v)[0]

    @validator('color_properties', pre=True, always=True)
    def _coerce_color_properties(cls, v):
        if v is None or (isinstance(v, dict) and len(v) == 0):
            return None
        return v

    @root_validator()
    def _validate_colors(cls, values):
        color_mode = values['color_mode']
        if color_mode == ColorMode.CYCLE:
            colors, values = _validate_cycle_mode(values)
        elif color_mode == ColorMode.COLORMAP:
            colors, values = _validate_colormap_mode(values)
        elif color_mode == ColorMode.DIRECT:
            colors = values['colors']

        # FIXME Local variable 'colors' might be referenced before assignment

        if values['current_color'] is None and len(colors) > 0:
            values['current_color'] = colors[-1]

        values['colors'] = colors
        return values

    def _set_color(
        self,
        color: ColorType,
        n_colors: int,
        properties: PropertyTable,
    ):
        """Set a color property. This is convenience function

        Parameters
        ----------
        color : (N, 4) array or str
            The value for setting edge or face_color
        n_colors : int
            The number of colors that needs to be set. Typically len(data).
        properties : PropertyTable
            The layer property values
        """
        # if the provided color is a string, first check if it is a key in the properties.
        # otherwise, assume it is the name of a color
        if is_color_mapped(color, properties):
            self.color_properties = properties[color]
            if guess_continuous(self.color_properties.values):
                self.color_mode = ColorMode.COLORMAP
            else:
                self.color_mode = ColorMode.CYCLE
        else:
            transformed_color = transform_color_with_defaults(
                num_entries=n_colors,
                colors=color,
                elem_name="color",
                default="white",
            )
            colors = normalize_and_broadcast_colors(
                n_colors, transformed_color
            )
            self.color_mode = ColorMode.DIRECT
            self.colors = colors

    def _refresh_colors(
        self,
        properties: PropertyTable,
        update_color_mapping: bool = False,
    ):
        """Calculate and update colors if using a cycle or color map
        Parameters
        ----------
        properties : Dict[str, np.ndarray]
           The layer properties to use to update the colors.
        update_color_mapping : bool
           If set to True, the function will recalculate the color cycle map
           or colormap (whichever is being used). If set to False, the function
           will use the current color cycle map or color map. For example, if you
           are adding/modifying points and want them to be colored with the same
           mapping as the other points (i.e., the new points shouldn't affect
           the color cycle map or colormap), set update_color_mapping=False.
           Default value is False.
        """
        if self.color_mode in [ColorMode.CYCLE, ColorMode.COLORMAP]:
            self.color_properties = properties[self.color_properties.name]
            if update_color_mapping is True:
                self.contrast_limits = None
            self.events.color_properties()

    def _add(
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
        if self.color_mode == ColorMode.DIRECT:
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
            if color is None:
                color = self.color_properties.default_value
            new_color_property_values = np.concatenate(
                (self.color_properties.values, np.repeat(color, n_colors)),
                axis=0,
            )
            self.color_properties = PropertyColumn.from_values(
                color_property_name, new_color_property_values
            )

            if update_clims and self.color_mode == ColorMode.COLORMAP:
                self.contrast_limits = None

    def _remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Remove the indicated color elements
        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        selected_indices = list(indices_to_remove)
        if len(selected_indices) > 0:
            if self.color_mode == ColorMode.DIRECT:
                self.colors = np.delete(self.colors, selected_indices, axis=0)
            else:
                self.color_properties = PropertyColumn.from_values(
                    self.color_properties.name,
                    np.delete(self.color_properties.values, selected_indices),
                )

    def _paste(self, colors: np.ndarray, properties: Dict[str, np.ndarray]):
        """Append colors to the ColorManager. Uses the color values if
        in direct mode and the properties in colormap or cycle mode.

        This method is for compatibility with the paste functionality
        in the layers.

        Parameters
        ----------
        colors : np.ndarray
            The (Nx4) color array of color values to add. These values are
            only used if the color mode is direct.
        properties : Dict[str, np.ndarray]
            The property values to add. These are used if the color mode
            is colormap or cycle.
        """
        if self.color_mode == ColorMode.DIRECT:
            self.colors = np.concatenate(
                (self.colors, transform_color(colors))
            )
        else:
            old_properties = self.color_properties.values
            new_properties = properties[self.color_properties.name]
            self.color_properties = PropertyColumn.from_values(
                self.color_properties.name,
                np.concatenate((old_properties, new_properties), axis=0),
            )

    def _update_properties(self, properties, name):
        if self.color_properties is not None:
            color_name = self.color_properties.name
            if color_name not in properties:
                self.color_mode = ColorMode.DIRECT
                self.color_properties = None
                warnings.warn(
                    trans._(
                        'property used for {name} dropped',
                        deferred=True,
                        name=name,
                    ),
                    RuntimeWarning,
                )
            else:
                # TODO: ideally this would not be necessary.
                self.color_properties = properties[color_name]

    def _update_current_properties(
        self, current_properties: Dict[str, np.ndarray]
    ):
        """This is updates the current_value of the color_properties when the
        layer current_properties is updated.

        This is a convenience method that is generally only called by the layer.

        Parameters
        ----------
        current_properties : Dict[str, np.ndarray]
            The new current property values
        """
        if self.color_properties is not None:
            property_name = self.color_properties.name
            if property_name in current_properties:
                new_current_value = np.squeeze(
                    current_properties[property_name]
                )
                self.color_properties.default_value = new_current_value

    def _update_current_color(
        self, current_color: np.ndarray, update_indices: Optional[list] = None
    ):
        """Update the current color and update the colors if requested.

        This is a convenience method and is generally called by the layer.

        Parameters
        ----------
        current_color : np.ndarray
            The new current color value.
        update_indices : list
            The indices of the color elements to update.
            If the list has length 0, no colors are updated.
            If the ColorManager is not in DIRECT mode, updating the values
            will change the mode to DIRECT.
        """
        if update_indices is None:
            update_indices = []
        self.current_color = transform_color(current_color)[0]
        if len(update_indices) > 0:
            self.color_mode = ColorMode.DIRECT
            cur_colors = self.colors.copy()
            cur_colors[update_indices] = self.current_color
            self.colors = cur_colors

    def _set_color_mode(
        self,
        property_table: PropertyTable,
        color_mode: Union[ColorMode, str],
        attribute: str,
    ):
        color_mode = ColorMode(color_mode)

        if color_mode == ColorMode.DIRECT:
            self.color_mode = color_mode
        elif color_mode in (ColorMode.CYCLE, ColorMode.COLORMAP):
            properties = property_table.all_values
            if self.color_properties is not None:
                color_property = self.color_properties.name
            else:
                color_property = ''
            if color_property == '':
                if properties:
                    new_color_property = next(iter(properties))
                    self.color_properties = property_table[new_color_property]
                    warnings.warn(
                        trans._(
                            '_{attribute}_color_property was not set, setting to: {new_color_property}',
                            deferred=True,
                            attribute=attribute,
                            new_color_property=new_color_property,
                        )
                    )
                else:
                    raise ValueError(
                        trans._(
                            'There must be a valid property to use {color_mode}',
                            deferred=True,
                            color_mode=color_mode,
                        )
                    )

            # ColorMode.COLORMAP can only be applied to numeric properties
            color_property = self.color_properties.name
            if (color_mode == ColorMode.COLORMAP) and not issubclass(
                properties[color_property].dtype.type, np.number
            ):
                raise TypeError(
                    trans._(
                        'selected property must be numeric to use ColorMode.COLORMAP',
                        deferred=True,
                    )
                )
            self.color_mode = color_mode

    @classmethod
    def _from_layer_kwargs(
        cls,
        colors: Union[dict, str, np.ndarray],
        properties: PropertyTable,
        n_colors: Optional[int] = None,
        continuous_colormap: Optional[Union[str, Colormap]] = None,
        contrast_limits: Optional[Tuple[float, float]] = None,
        categorical_colormap: Optional[
            Union[CategoricalColormap, list, np.ndarray]
        ] = None,
        color_mode: Optional[Union[ColorMode, str]] = None,
        current_color: Optional[np.ndarray] = None,
        default_color_cycle: np.ndarray = np.array([1, 1, 1, 1]),
    ):
        """Initialize a ColorManager object from layer kwargs. This is a convenience
        function to coerce possible inputs into ColorManager kwargs

        """
        color_values = colors
        color_properties = None

        if isinstance(colors, dict):
            # if the kwargs are passed as a dictionary, unpack them
            color_values = colors.get('colors', None)
            current_color = colors.get('current_color', current_color)
            color_mode = colors.get('color_mode', color_mode)
            color_properties = colors.get('color_properties', None)
            continuous_colormap = colors.get(
                'continuous_colormap', continuous_colormap
            )
            contrast_limits = colors.get('contrast_limits', contrast_limits)
            categorical_colormap = colors.get(
                'categorical_colormap', categorical_colormap
            )

            if isinstance(color_properties, str):
                try:
                    color_properties = properties[color_properties]
                except KeyError:
                    raise KeyError(
                        trans._(
                            'if color_properties is a string, it should be a property name',
                            deferred=True,
                        )
                    )

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
                color_properties = properties[color_values]
                if color_mode is None:
                    if guess_continuous(color_properties.values):
                        color_mode = ColorMode.COLORMAP
                    else:
                        color_mode = ColorMode.CYCLE

                color_kwargs.update(
                    {
                        'color_mode': color_mode,
                        'color_properties': color_properties,
                    }
                )

            else:
                # direct mode
                if n_colors == 0:
                    if current_color is None:
                        current_color = transform_color(color_values)[0]
                    color_kwargs.update(
                        {
                            'color_mode': ColorMode.DIRECT,
                            'current_color': current_color,
                        }
                    )
                else:
                    transformed_color = transform_color_with_defaults(
                        num_entries=n_colors,
                        colors=color_values,
                        elem_name="colors",
                        default="white",
                    )
                    colors = normalize_and_broadcast_colors(
                        n_colors, transformed_color
                    )
                    color_kwargs.update(
                        {'color_mode': ColorMode.DIRECT, 'colors': colors}
                    )
        else:
            color_kwargs.update(
                {
                    'color_mode': color_mode,
                    'color_properties': color_properties,
                }
            )

        return cls(**color_kwargs)
