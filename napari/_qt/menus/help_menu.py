import webbrowser
from typing import TYPE_CHECKING

from packaging.version import parse
from qtpy.QtWidgets import QAction

from napari import __version__

from ...utils.translations import trans
from ..dialogs.qt_about import QtAbout
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window

try:
    from napari_error_reporter import ask_opt_in
except ModuleNotFoundError:
    ask_opt_in = None

# Get the version for proper links to documentation
v = parse(__version__)
VERSION = "dev" if v.is_devrelease else str(v)


class HelpMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        super().__init__(trans._('&Help'), window._qt_window)
        ACTIONS = [
            {
                'text': trans._('Getting started'),
                'slot': lambda e: webbrowser.open(
                    f'https://napari.org/{VERSION}/tutorials/start_index.html'
                ),
                'statusTip': trans._('Open Getting started webpage'),
            },
            {
                'text': trans._('Tutorials'),
                'slot': lambda e: webbrowser.open(
                    f'https://napari.org/{VERSION}/tutorials/index.html'
                ),
                'statusTip': trans._('Open Tutorials webpage'),
            },
            {
                'text': trans._('Examples Gallery'),
                'slot': lambda e: webbrowser.open(
                    f'https://napari.org/{VERSION}/gallery.html'
                ),
                'statusTip': trans._('Open Examples Gallery webpage'),
            },
            {
                'when': VERSION != "dev",
                'text': trans._('Release Notes'),
                'slot': lambda e: webbrowser.open(
                    f'https://napari.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html'
                ),
                'statusTip': trans._('Open Release Notes webpage'),
            },
            {
                'text': trans._('napari homepage'),
                'slot': lambda e: webbrowser.open('https://napari.org'),
                'statusTip': trans._('Open napari.org webpage'),
            },
            {},
            {
                'text': trans._('napari info'),
                'slot': lambda e: QtAbout.showAbout(window._qt_window),
                'shortcut': 'Ctrl+/',
                'statusTip': trans._('About napari'),
                # on macOS this will be properly placed in the napari menu as:
                # About napari
                'menuRole': QAction.AboutRole,
            },
        ]
        if ask_opt_in is not None:
            ACTIONS.append(
                {
                    'text': trans._('Bug reporting opt in/out...'),
                    'slot': lambda: ask_opt_in(force=True),
                }
            )

        populate_menu(self, ACTIONS)
