import os
from pathlib import Path

from ..settings import get_settings


def update_open_history(filename):
    """Updates open history of files in settings.

    Parameters
    ----------
    filename : str
        New file being added to open history.
    """
    settings = get_settings()
    folders = settings.application.open_history
    new_loc = os.path.dirname(filename)
    if new_loc in folders:
        folders.insert(0, folders.pop(folders.index(new_loc)))
    else:
        folders.insert(0, new_loc)

    folders = folders[0:10]
    settings.application.open_history = folders


def update_save_history(filename):
    """Updates save history of files in settings.

    Parameters
    ----------
    filename : str
        New file being added to save history.
    """
    settings = get_settings()
    folders = settings.application.save_history
    new_loc = os.path.dirname(filename)
    if new_loc in folders:
        folders.insert(0, folders.pop(folders.index(new_loc)))
    else:
        folders.insert(0, new_loc)

    folders = folders[0:10]
    settings.application.save_history = folders


def get_open_history():
    """A helper for history handling."""
    settings = get_settings()
    folders = settings.application.open_history
    folders = [f for f in folders if os.path.isdir(f)]
    return folders or [str(Path.home())]


def get_save_history():
    """A helper for history handling."""
    settings = get_settings()
    folders = settings.application.save_history
    folders = [f for f in folders if os.path.isdir(f)]
    return folders or [str(Path.home())]
