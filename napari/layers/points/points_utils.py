from typing import Dict, Tuple, Union

import numpy as np
from vispy.color.colormap import Colormap


def dataframe_to_properties(dataframe) -> Dict[str, np.ndarray]:
    """Convert a dataframe to Points.properties formatted dictionary.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe object to be converted to a properties dictionary

    Returns
    -------
    dict[str, np.ndarray]
        A properties dictionary where the key is the property name and the value
        is an ndarray with the property value for each point.
    """

    properties = {col: np.asarray(dataframe[col]) for col in dataframe}
    return properties


def guess_continuous(property: np.ndarray) -> bool:
    """Guess if the property is continuous (return True) or categorical (return False)"""
    # if the property is a floating type, guess continuous
    if issubclass(property.dtype.type, np.floating) and len(property < 16):
        return True
    else:
        return False


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
    colormap : vispy.color.Colormap
        The vispy colormap object to apply to the property
    contrast_limits: Union[None, Tuple[float, float]]
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
