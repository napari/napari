"""Utilities that provide magic functionality.

Set the ``MAGICNAME`` environment variable to a none-null value to enable magic naming.
"""

import inspect

from collections import ChainMap
import os


def magic_name(value, *, level=1):
    """Fetch the name of the variable with the given value passed to the calling function.

    Parameters
    ----------
    value : any
        The value of the desired variable.
    level : int, kwonly, optional
        The level of nestedness to traverse.
    """
    if level < 1:
        raise ValueError('cannot have a level lower than 1')

    if not os.getenv('MAGICNAME'):
        return

    level += 1
    frame = inspect.currentframe()

    for i in range(level):
        frame = frame.f_back

    code = frame.f_code

    varmap = ChainMap(frame.f_locals, frame.f_globals)
    names = *code.co_varnames, *code.co_names

    for name in names:
        if name.isidentifier() and name in varmap and varmap[name] is value:
            return name
