import platform
from functools import partial
from typing import TYPE_CHECKING

from ...settings import get_settings
from ...utils import config as async_config
from ...utils.translations import trans
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class ViewMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&View'), window._qt_window)

        def _toggle_dict(text, name, prop):
            # helper func to make a Action dict for togglers
            obj = getattr(window._qt_viewer.viewer, name)
            return {
                'text': text,
                'slot': partial(setattr, obj, prop),
                'checkable': True,
                'checked': getattr(obj, prop),
                'check_on': getattr(obj.events, prop),
            }

        settings = get_settings()

        ACTIONS = [
            {
                'text': trans._('Toggle Full Screen'),
                'slot': window._toggle_fullscreen,
                'shortcut': 'Ctrl+F',
            },
            {
                'when': platform.system() != "Darwin",
                'text': trans._('Toggle Menubar Visibility'),
                'slot': window._toggle_menubar_visible,
                'shortcut': 'Ctrl+M',
                'statusTip': trans._('Hide Menubar'),
            },
            {
                'text': trans._('Toggle Play'),
                'slot': window._toggle_play,
                'shortcut': 'Ctrl+Alt+P',
            },
            {},
            {
                'when': async_config.async_octree,
                'text': trans._('Toggle Chunk Outlines'),
                'slot': window._qt_viewer._toggle_chunk_outlines,
                'shortcut': 'Ctrl+Alt+O',
            },
            {
                'menu': 'Axes',
                'items': [
                    _toggle_dict(trans._('Visible'), 'axes', 'visible'),
                    _toggle_dict(trans._('Colored'), 'axes', 'colored'),
                    _toggle_dict(trans._('Labels'), 'axes', 'labels'),
                    _toggle_dict(trans._('Dashed'), 'axes', 'dashed'),
                    _toggle_dict(trans._('Arrows'), 'axes', 'arrows'),
                ],
            },
            {
                'menu': 'Scale Bar',
                'items': [
                    _toggle_dict(trans._('Visible'), 'scale_bar', 'visible'),
                    _toggle_dict(trans._('Colored'), 'scale_bar', 'colored'),
                    _toggle_dict(trans._('Ticks'), 'scale_bar', 'ticks'),
                ],
            },
            {
                'text': trans._('Layer Tooltip visibility'),
                'slot': self._tooltip_visibility_toggle,
                'checkable': True,
                'checked': settings.appearance.layer_tooltip_visibility,
                'check_on': settings.appearance.events.layer_tooltip_visibility,
            },
            {
                'text': trans._('Activity Dock'),
                'slot': window._status_bar._toggle_activity_dock,
                'checkable': True,
                'checked': window._qt_window._activity_dialog.isVisible(),
                'check_on': window._status_bar._activity_item._activityBtn.toggled,
            },
            {},
        ]

        populate_menu(self, ACTIONS)

    def _tooltip_visibility_toggle(self, value):
        get_settings().appearance.layer_tooltip_visibility = value
