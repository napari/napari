from typing import Tuple, Union

import numpy as np

from napari.utils import Colormap


def guess_continuous(property: np.ndarray) -> bool:
    """Guess if the property is continuous (return True) or categorical (return False)

    The property is guessed as continuous if it is a float or contains over 16 elements.

    Parameters
    ----------
    property : np.ndarray
        The property values to guess if they are continuous

    Returns
    -------
    continuous : bool
        True of the property is guessed to be continuous, False if not.
    """
    # if the property is a floating type, guess continuous
    if (
        issubclass(property.dtype.type, np.floating)
        or len(np.unique(property)) > 16
    ):
        return True
    else:
        return False


def is_color_mapped(color, properties):
    """ determines if the new color argument is for directly setting or cycle/colormap"""
    if isinstance(color, str):
        if color in properties:
            return True
        else:
            return False
    elif isinstance(color, dict):
        return True
    elif isinstance(color, (list, np.ndarray)):
        return False
    else:
        raise ValueError(
            'face_color should be the name of a color, an array of colors, or the name of an property'
        )


def map_property(
    prop: np.ndarray,
    colormap: Colormap,
    contrast_limits: Union[None, Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
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
