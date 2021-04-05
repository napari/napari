import os

from ..utils.settings import SETTINGS


def update_open_history(filename):
    """Updates open history of files in settings."""
    folders = SETTINGS.application.open_history
    new_loc = os.path.dirname(filename)
    if new_loc in folders:
        folders.insert(0, folders.pop(folders.index(new_loc)))
    else:
        folders.insert(0, new_loc)
    folders = folders[0:10]
    SETTINGS.application.open_history = folders
    # SETTINGS.application.last_opened_dir = folders[0]


def update_save_history(filename):
    """Updates save history of files in settings."""
    folders = SETTINGS.application.save_history
    new_loc = os.path.dirname(filename)
    if new_loc in folders:
        folders.insert(0, folders.pop(folders.index(new_loc)))
    else:
        folders.insert(0, new_loc)
    folders = folders[0:10]
    SETTINGS.application.save_history = folders
    # SETTINGS.application.last_saved_dir = folders[0]


def get_open_history():
    """A helper for history handling."""
    folders = SETTINGS.application.open_history
    folders = [f for f in folders if os.path.isdir(f)]
    return folders


def get_save_history():
    """A helper for history handling."""
    folders = SETTINGS.application.save_history
    folders = [f for f in folders if os.path.isdir(f)]
    return folders
