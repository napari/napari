"""Actions related to the 'Help' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_help.py`.
"""
import webbrowser
from typing import List

from app_model.types import Action
from packaging.version import parse

from napari import __version__
from napari._app_model.constants import CommandId, MenuId

v = parse(__version__)
VERSION = "dev" if v.is_devrelease else str(v)


def _open_getting_started():
    webbrowser.open(f'https://napari.org/{VERSION}/tutorials/start_index.html')


HELP_ACTIONS: List[Action] = [
    Action(
        id=CommandId.NAPARI_GETTING_STARTED,
        title=CommandId.NAPARI_GETTING_STARTED.title,
        callback=_open_getting_started,
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
]
