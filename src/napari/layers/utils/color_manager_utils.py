from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari.utils.colormaps import Colormap
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers.utils.color_manager import ColorManager


def guess_continuous(color_map: np.ndarray) -> bool:
    """Guess if the property is continuous (return True) or categorical (return False)

    The property is guessed as continuous if it is a float or contains over 16 elements.

    Parameters
    ----------
    color_map : np.ndarray
        The property values to guess if they are continuous

    Returns
    -------
    continuous : bool
        True of the property is guessed to be continuous, False if not.
    """
    # if the property is a floating type, guess continuous
    return issubclass(color_map.dtype.type, np.floating) or (
        len(np.unique(color_map)) > 16
        and issubclass(color_map.dtype.type, np.integer)
    )


def is_color_mapped(color, properties):
    """determines if the new color argument is for directly setting or cycle/colormap"""
    if isinstance(color, str):
        return color in properties
    if isinstance(color, dict):
        return True
    if isinstance(color, list | np.ndarray):
        return False

    raise ValueError(
        trans._(
            'face_color should be the name of a color, an array of colors, or the name of an property',
            deferred=True,
        )
    )


def map_property(
    prop: np.ndarray,
    colormap: Colormap,
    contrast_limits: None | tuple[float, float] = None,
) -> tuple[Array, tuple[float, float]]:
    """Apply a colormap to a property

    Parameters
    ----------
    prop : np.ndarray
        The property to be colormapped
    colormap : napari.utils.Colormap
        The colormap object to apply to the property
    contrast_limits : Union[None, Tuple[float, float]]
        The contrast limits for applying the colormap to the property.
        If a 2-tuple is provided, it should be provided as (lower_bound, upper_bound).
        If None is provided, the contrast limits will be set to (property.min(), property.max()).
        Default value is None.
    """

    if contrast_limits is None:
        contrast_limits = (prop.min(), prop.max())
    normalized_properties = np.interp(prop, contrast_limits, (0, 1))
    mapped_properties = colormap.map(normalized_properties)

    return mapped_properties, contrast_limits


def _validate_colormap_mode(color_manager: ColorManager) -> None:
    """Validate the ColorManager field values specific for colormap mode
    This is called by the root_validator in ColorManager

    Parameters
    ----------
    values : dict
        The field values that are passed to the ColorManager root validator

    Returns
    -------
    colors : np.ndarray
        The (Nx4) color array to set as ColorManager.colors
    values : dict
    """
    if color_manager.color_properties is None:
        raise ValueError(
            'color properties must be set to validate colormap mode'
        )
    color_properties = color_manager.color_properties
    cmap = color_manager.continuous_colormap
    if len(color_properties.values) > 0:
        if color_manager.contrast_limits is None:
            color_manager.colors, color_manager.contrast_limits = map_property(
                prop=color_properties.values,
                colormap=cmap,
            )
        else:
            color_manager.colors, _ = map_property(
                prop=color_properties.values,
                colormap=cmap,
                contrast_limits=color_manager.contrast_limits,
            )
    else:
        color_manager.colors = np.empty((0, 4))  # type: ignore
        current_prop_value = color_properties.current_value
        if current_prop_value is not None:
            color_manager.current_color = cmap.map(current_prop_value)[0]

    if len(color_manager.colors) == 0:
        color_manager.colors = np.empty((0, 4))  # type: ignore


def _validate_cycle_mode(
    color_manager: ColorManager,
) -> None:
    """Validate the ColorManager field values specific for color cycle mode
    This is called by the root_validator in ColorManager

    Parameters
    ----------
    values : dict
        The field values that are passed to the ColorManager root validator

    Returns
    -------
    colors : np.ndarray
        The (Nx4) color array to set as ColorManager.colors
    values : dict
    """
    color_properties = color_manager.color_properties
    if color_properties is None:
        raise ValueError(
            'color properties must be set to validate colormap mode'
        )
    cmap = color_manager.categorical_colormap
    if len(color_properties.values) == 0:
        color_manager.colors = np.empty((0, 4))  # type: ignore
        current_prop_value = color_properties.current_value
        if current_prop_value is not None:
            color_manager.current_color = cmap.map(current_prop_value)[0]
    else:
        color_manager.colors = cmap.map(color_properties.values)  # type: ignore
