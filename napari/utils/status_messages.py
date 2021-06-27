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
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        # FIMXE: use an f-string?
        return '[' + str.join(', ', [status_format(v) for v in value]) + ']'
    if value is None:
        return ''
    if isinstance(value, float) or np.issubdtype(type(value), np.floating):
        return format_float(value)
    elif isinstance(value, int) or np.issubdtype(type(value), np.integer):
        return str(value)
    else:
        return str(value)


def generate_layer_status(name, position, value):
    """Generate a status message based on the coordinates and value

    Parameters
    ----------
    name : str
        Name of the layer.
    position : tuple or list
        List of coordinates, say of the cursor.
    value : Any
        The value to be printed.

    Returns
    -------
    msg : string
        String containing a message that can be used as a status update.
    """
    full_coord = np.round(position).astype(int)

    msg = f'{name} {full_coord}'

    if value is not None:
        if isinstance(value, tuple) and value != (None, None):
            # it's a multiscale -> value = (data_level, value)
            msg += f': {status_format(value[0])}'
            if value[1] is not None:
                msg += f', {status_format(value[1])}'
        else:
            # it's either a grayscale or rgb image (scalar or list)
            msg += f': {status_format(value)}'
    return msg
