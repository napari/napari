from collections.abc import Iterable

import numpy as np


def format_float(value):
    """Nice float formatting into strings."""
    return f'{value:0.3g}'


def status_format(value):
    """Return a "nice" string representation of a value.

    Parameters
    ----------
    value : Any
        The value to be printed.

    Returns
    -------
    formatted : str
        The string resulting from formatting.

    Examples
    --------
    >>> values = np.array([1, 10, 100, 1000, 1e6, 6.283, 123.932021,
    ...                    1123.9392001, 2 * np.pi, np.exp(1)])
    >>> status_format(values)
    '[1, 10, 100, 1e+03, 1e+06, 6.28, 124, 1.12e+03, 6.28, 2.72]'
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
