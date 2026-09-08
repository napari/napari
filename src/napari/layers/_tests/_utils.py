import numpy as np


def compare_dicts(dict1, dict2):
    """
    The compare_dicts method compares two dictionaries for equality.

    This is mainly used to allow for layer.data.events tests in order to avoid comparison of 2 arrays.

    dict1
        dict to be compared to other dict2
    dict2
        dict to be compared to other dict1

    Returns
    -------
    bool
        Whether the two dictionaries are equal
    """
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        elif val1 != val2:
            return False

    return True
