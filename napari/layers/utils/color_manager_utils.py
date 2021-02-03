import numpy as np


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
