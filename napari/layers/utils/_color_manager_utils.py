import numpy as np


def is_color_mapped(color, properties):
    """ determines if the new color argument is for directly setting or cycle/colormap

    Parameters
    ----------
    color : (N, 4) array or str
        The color argument to evaluate if it is a color or a properties
        key to map against
    properties : dict
        The layer properties to check if the color argument is a valid key.

    Returns
    -------
    color_mapped : bool
        Set to True if the color value is a color mapping and set to false
        when the color value is itself a color.

    """
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
        raise TypeError(
            'color should be the name of a color, an array of colors, or the name of an property'
        )
