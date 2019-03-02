"""Automatically generate names.
"""
import re
from .misc import formatdoc


sep = ' '
start = 1

numbered_patt = re.compile(r'(?<!\d)(?:\d+|)$')


def _inc_name_count_sub(match):
    count = match.group(0)

    try:
        count = int(count)
    except ValueError:  # not an int
        count = f'{sep}{start}'
    else:
        count = f'{count + 1}'

    return count


@formatdoc
def inc_name_count(name):
    """Increase a name's count matching `{numbered_patt}` by ``1``.

    If the name is not already numbered, append '{sep}{start}'.

    Parameters
    ----------
    name : str
        Original name.

    Returns
    -------
    incremented_name : str
        Numbered name incremented by ``1``.
    """
    return numbered_patt.sub(_inc_name_count_sub, name)
