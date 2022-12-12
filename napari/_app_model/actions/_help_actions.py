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

HELP_URLS = {
    "getting_started": f'https://napari.org/{VERSION}/tutorials/start_index.html',
    "tutorials": f'https://napari.org/{VERSION}/tutorials/index.html',
    "layers_guide": f'https://napari.org/{VERSION}/howtos/layers/index.html',
    "examples_gallery": f'https://napari.org/{VERSION}/gallery.html',
    "release_notes": f'https://napari.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html',
    "github_issue": 'https://github.com/napari/napari/issues',
    "homepage": 'https://napari.org',
}

HELP_ACTIONS: List[Action] = [
    Action(
        id=CommandId.NAPARI_HOMEPAGE,
        title=CommandId.NAPARI_HOMEPAGE.title,
        callback=lambda: webbrowser.open(HELP_URLS['homepage']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_GETTING_STARTED,
        title=CommandId.NAPARI_GETTING_STARTED.title,
        callback=lambda: webbrowser.open(HELP_URLS['getting_started']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_TUTORIALS,
        title=CommandId.NAPARI_TUTORIALS.title,
        callback=lambda: webbrowser.open(HELP_URLS['tutorials']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_LAYERS_GUIDE,
        title=CommandId.NAPARI_LAYERS_GUIDE.title,
        callback=lambda: webbrowser.open(HELP_URLS['layers_guide']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_EXAMPLES,
        title=CommandId.NAPARI_EXAMPLES.title,
        callback=lambda: webbrowser.open(HELP_URLS['examples_gallery']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id=CommandId.NAPARI_RELEASE_NOTES,
        title=CommandId.NAPARI_RELEASE_NOTES.title,
        callback=lambda: webbrowser.open(
            HELP_URLS['release_notes'],
        ),
        menus=[{'id': MenuId.MENUBAR_HELP, 'when': VERSION != "dev"}],
    ),
    Action(
        id=CommandId.NAPARI_GITHUB_ISSUE,
        title=CommandId.NAPARI_GITHUB_ISSUE.title,
        callback=lambda: webbrowser.open(
            HELP_URLS['github_issue'],
        ),
        menus=[{'id': MenuId.MENUBAR_HELP, 'when': VERSION == "dev"}],
    ),
]
