from collections import Iterable
import numpy as np


def format_float(value):
    return f'{value:0.3}'


def status_format(value):
    """Return a "nice" string representation of a value.

    Parameters
    ----------
    value : any type
        The value to be printed.

    Returns
    -------
    formatted : str
        The string resulting from formatting.
    """
    if isinstance(value, Iterable):
        return '[' + str.join(', ', [status_format(v) for v in value]) + ']'
    if value is None:
        return ''
    if isinstance(value, float) or np.issubdtype(type(value), np.floating):
        return format_float(value)
    elif isinstance(value, int) or np.issubdtype(type(value), np.integer):
        return str(value)
    else:
        return str(value)
