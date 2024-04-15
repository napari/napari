"""Actions related to the 'Help' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_help.py`.
"""

import webbrowser

from app_model.types import Action
from packaging.version import parse

from napari import __version__
from napari._app_model.constants import MenuGroup, MenuId
from napari.utils.translations import trans

v = parse(__version__)
VERSION = 'dev' if v.is_devrelease else str(v)

HELP_URLS = {
    'getting_started': f'https://napari.org/{VERSION}/tutorials/start_index.html',
    'tutorials': f'https://napari.org/{VERSION}/tutorials/index.html',
    'layers_guide': f'https://napari.org/{VERSION}/howtos/layers/index.html',
    'examples_gallery': f'https://napari.org/{VERSION}/gallery.html',
    'release_notes': f'https://napari.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html',
    'github_issue': 'https://github.com/napari/napari/issues',
    'homepage': 'https://napari.org',
}

HELP_ACTIONS: list[Action] = [
    Action(
        id='napari.window.help.getting_started',
        title=trans._('Getting started'),
        callback=lambda: webbrowser.open(HELP_URLS['getting_started']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.tutorials',
        title=trans._('Tutorials'),
        callback=lambda: webbrowser.open(HELP_URLS['tutorials']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.layers_guide',
        title=trans._('Using Layers Guides'),
        callback=lambda: webbrowser.open(HELP_URLS['layers_guide']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.examples',
        title=trans._('Examples Gallery'),
        callback=lambda: webbrowser.open(HELP_URLS['examples_gallery']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.release_notes',
        title=trans._('Release Notes'),
        callback=lambda: webbrowser.open(
            HELP_URLS['release_notes'],
        ),
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'when': VERSION != 'dev',
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id='napari.window.help.github_issue',
        title=trans._('Report an issue on GitHub'),
        callback=lambda: webbrowser.open(
            HELP_URLS['github_issue'],
        ),
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'when': VERSION == 'dev',
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id='napari.window.help.homepage',
        title=trans._('napari homepage'),
        callback=lambda: webbrowser.open(HELP_URLS['homepage']),
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.NAVIGATION}],
    ),
]
