from dataclasses import InitVar
from typing import Optional, Tuple, Union

import numpy as np

from ...utils.colormaps import CategoricalColormap, Colormap, ensure_colormap
from ...utils.colormaps.color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color_with_defaults,
)
from ...utils.events.dataclass import Property, evented_dataclass
from ._color_manager_constants import ColorMode
from ._color_manager_utils import is_color_mapped
from .layer_utils import guess_continuous, map_property


@evented_dataclass(events=True, properties=True)
class ColorManager:
    """Colors for a display property

    Parameters
    ----------
    values : np.ndarray
        The RGBA color for each data entry
    mode : ColorMode
        Color setting mode. Should be one of the following:
        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute

    """

    colors: InitVar[Optional[Union[str, np.ndarray, list]]] = None
    n_colors: InitVar[int] = 0
    properties: InitVar[Optional[dict]] = None

    values: Optional[np.ndarray] = None
    mode: Property[ColorMode, str, None] = ColorMode.DIRECT
    color_property: str = ''
    continuous_colormap: Property[Colormap, None, ensure_colormap] = 'viridis'
    continuous_contrast_limits: Optional[Tuple[float, float]] = None
    categorical_colormap: Property[
        CategoricalColormap, None, CategoricalColormap
    ] = np.array([[0, 0, 0, 1]])

    def __post_init__(self, colors, n_colors, properties):
        if colors is None:
            colors = np.empty((0, 4))
        self.set_color(color=colors, n_colors=n_colors, properties=properties)

    def set_color(self, color: ColorType, n_colors: int, properties: dict):
        """ Set the face_color or edge_color property

        Parameters
        ----------
        color : (N, 4) array or str
            The new color. If an array, color should be an
            Nx4 RGBA array for N colors or a 1x4 RGBA array
            that gets broadcast to N colors.
        n_colors:
            The total number of colors that should be created.
        """

        if is_color_mapped(color, properties):
            if guess_continuous(properties[color]):
                self._mode = ColorMode.COLORMAP
            else:
                self._mode = ColorMode.CYCLE
            self._color_property = color
            if n_colors == 0:
                self.values = np.empty((0, 4))
            else:
                self.refresh_colors(properties=properties)
        else:
            if n_colors == 0:
                self.values = np.empty((0, 4))
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
                self.values = colors
            self._color_mode = ColorMode.DIRECT

    def add(self, color, n_colors: int = 1):
        if self._mode == ColorMode.DIRECT:
            new_color = color
        elif self._mode == ColorMode.CYCLE:
            if isinstance(color, str):
                color = [color]
            new_color = self.categorical_colormap.map(color)
        elif self._mode == ColorMode.COLORMAP:
            new_color = self.continuous_colormap.map(color)
        transformed_color = transform_color_with_defaults(
            num_entries=n_colors,
            colors=new_color,
            elem_name="color",
            default="white",
        )
        broadcasted_colors = normalize_and_broadcast_colors(
            n_colors, transformed_color
        )
        self._values = np.concatenate((self.values, broadcasted_colors))

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Remove the indicated color elements

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        selected_indices = list(indices_to_remove)
        if len(selected_indices) > 0:
            self._values = np.delete(self.values, selected_indices, axis=0)

    def refresh_colors(
        self, properties: dict, update_color_mapping: bool = False
    ):
        """Calculate and update face or edge colors if using a cycle or color map

        Parameters
        ----------
        properties : dict
            The layer properties to map the colors against.
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying points and want them to be colored with the same
            mapping as the other points (i.e., the new points shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.
        """

        if self._mode == ColorMode.CYCLE:
            color_properties = properties[self.color_property]

            colors = self.categorical_colormap.map(color_properties)

        elif self._mode == ColorMode.COLORMAP:

            color_properties = properties[self.color_property]
            if len(color_properties) > 0:
                if (
                    update_color_mapping
                    or self.continuous_contrast_limits is None
                ):

                    colors, contrast_limits = map_property(
                        prop=color_properties,
                        colormap=self.continuous_colormap,
                    )
                    self.continuous_contrast_limits = contrast_limits
                else:
                    colors, _ = map_property(
                        prop=color_properties,
                        colormap=self.continuous_colormap,
                        contrast_limits=self.continuous_contrast_limits,
                    )
            else:
                colors = np.empty((0, 4))
        if len(colors) == 0:
            colors = np.empty((0, 4))
        self.values = colors
        self.events.values()
