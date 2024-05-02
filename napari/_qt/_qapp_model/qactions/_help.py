"""Actions related to the 'Help' menu that require Qt.

'Help' actions that do not require Qt should go in a new '_help_actions.py'
file within `napari/_app_model/actions/`.
"""

import sys
from functools import partial
from typing import Callable
from webbrowser import open

from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod
from packaging.version import parse

from napari import __version__
from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.dialogs.qt_about import QtAbout
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans

try:
    from napari_error_reporter import ask_opt_in
except ModuleNotFoundError:
    ask_opt_in = None


def _show_about(window: Window):
    QtAbout.showAbout(window._qt_window)


v = parse(__version__)
VERSION = 'dev' if v.is_devrelease else str(v)

HELP_FUNCS: dict[str, Callable] = {
    'getting_started': partial(
        open, url=f'https://napari.org/{VERSION}/tutorials/start_index.html'
    ),
    'tutorials': partial(
        open, url=f'https://napari.org/{VERSION}/tutorials/index.html'
    ),
    'layers_guide': partial(
        open, url=f'https://napari.org/{VERSION}/howtos/layers/index.html'
    ),
    'examples_gallery': partial(
        open, url=f'https://napari.org/{VERSION}/gallery.html'
    ),
    'release_notes': partial(
        open,
        url=f'https://napari.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html',
    ),
    'github_issue': partial(
        open, url='https://github.com/napari/napari/issues'
    ),
    'homepage': partial(open, url='https://napari.org'),
}

Q_HELP_ACTIONS: list[Action] = [
    Action(
        id='napari.window.help.info',
        title=trans._('‎napari Info'),
        callback=_show_about,
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.RENDER}],
        status_tip=trans._('About napari'),
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.Slash)],
    ),
    Action(
        id='napari.window.help.about_macos',
        title=trans._('About napari'),
        callback=_show_about,
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'group': MenuGroup.RENDER,
                'when': sys.platform == 'darwin',
            }
        ],
        status_tip=trans._('About napari'),
    ),
    Action(
        id='napari.window.help.getting_started',
        title=trans._('Getting started'),
        callback=HELP_FUNCS['getting_started'],
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.tutorials',
        title=trans._('Tutorials'),
        callback=HELP_FUNCS['tutorials'],
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.layers_guide',
        title=trans._('Using Layers Guides'),
        callback=HELP_FUNCS['layers_guide'],
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.examples',
        title=trans._('Examples Gallery'),
        callback=HELP_FUNCS['examples_gallery'],
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.release_notes',
        title=trans._('Release Notes'),
        callback=HELP_FUNCS['release_notes'],
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
        callback=HELP_FUNCS['github_issue'],
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
        callback=HELP_FUNCS['homepage'],
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.NAVIGATION}],
    ),
]

if ask_opt_in is not None:
    Q_HELP_ACTIONS.append(
        Action(
            id='napari.window.help.bug_report_opt_in',
            title=trans._('Bug Reporting Opt In/Out...'),
            callback=lambda: ask_opt_in(force=True),
            menus=[{'id': MenuId.MENUBAR_HELP}],
        )
    )
