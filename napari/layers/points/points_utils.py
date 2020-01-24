from typing import Dict, Tuple, Union

import numpy as np
from vispy.color.colormap import Colormap


def dataframe_to_properties(dataframe) -> Dict[str, np.ndarray]:
    """ converts a dataframe to Points.properties formatted dictionary"""

    properties = {col: np.asarray(dataframe[col]) for col in dataframe}
    return properties


def guess_continuous(property: np.ndarray) -> bool:
    """guess if the property is continuous (return True) or categorical (return False)"""
    # if the property is a floating type, guess continuous
    if issubclass(property.dtype.type, np.floating):
        return True
    else:
        return False


def map_properties(
    properties: np.ndarray,
    colormap: Colormap,
    contrast_limits: Union[None, Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:

    if contrast_limits is None:
        contrast_limits = (properties.min(), properties.max())
    normalized_properties = np.interp(properties, contrast_limits, (0, 1))
    mapped_properties = colormap.map(normalized_properties)

    return mapped_properties, contrast_limits
