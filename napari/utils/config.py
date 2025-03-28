"""Napari Configuration."""

import os


def _set(env_var: str) -> bool:
    """Return True if the env variable is set and non-zero.

    Returns
    -------
    bool
        True if the env var was set to a non-zero value.
    """
    return os.getenv(env_var) not in [None, '0']


"""
Experimental Features

Shared Memory Server
--------------------
Experimental shared memory service. Only enabled if NAPARI_MON is set to
the path of a config file. See this PR for more info:
https://github.com/napari/napari/pull/1909.
"""

# Shared Memory Server
monitor = _set('NAPARI_MON')
