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


HELP_ACTIONS: List[Action] = [
    Action(
        id=CommandId.NAPARI_GETTING_STARTED,
        title=CommandId.NAPARI_GETTING_STARTED.title,
        callback=lambda: webbrowser.open(
            f'https://napari.org/{VERSION}/tutorials/start_index.html'
        ),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_TUTORIALS,
        title=CommandId.NAPARI_TUTORIALS.title,
        callback=lambda: webbrowser.open(
            f'https://napari.org/{VERSION}/tutorials/index.html'
        ),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_LAYERS_GUIDE,
        title=CommandId.NAPARI_LAYERS_GUIDE.title,
        callback=lambda: webbrowser.open(
            f'https://napari.org/{VERSION}/howtos/layers/index.html'
        ),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_EXAMPLES,
        title=CommandId.NAPARI_EXAMPLES.title,
        callback=lambda: webbrowser.open(
            f'https://napari.org/{VERSION}/gallery.html'
        ),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_RELEASE_NOTES,
        title=CommandId.NAPARI_RELEASE_NOTES.title,
        callback=lambda: webbrowser.open(
            f'https://napari.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html'
        ),
        menus=[{'id': MenuId.MENUBAR_HELP}],
        enablement=VERSION != "dev",
    ),
    Action(
        id=CommandId.NAPARI_HOMEPAGE,
        title=CommandId.NAPARI_HOMEPAGE.title,
        callback=lambda: webbrowser.open('https://napari.org'),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
]
