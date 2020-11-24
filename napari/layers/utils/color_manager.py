import numpy as np

from ...utils.events.dataclass import Property, dataclass
from ..utils.color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color_with_defaults,
)
from ._color_manager_constants import ColorMode


@dataclass(events=True, properties=True)
class ColorManager:
    """Colors for a display property

    Parameters
    ----------
    colors : np.ndarray
        The RGBA color for each data entry
    mode : ColorMode


    """

    colors: np.ndarray = np.empty((0, 4))
    mode: Property[ColorMode, str, None] = ColorMode.DIRECT

    def set_color(self, color: ColorType, n_colors: int):
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
        transformed_color = transform_color_with_defaults(
            num_entries=n_colors,
            colors=color,
            elem_name="face_color",
            default="white",
        )
        colors = normalize_and_broadcast_colors(n_colors, transformed_color)
        self.colors = colors
