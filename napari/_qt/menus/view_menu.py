from functools import partial
from typing import TYPE_CHECKING

from ...utils.translations import trans
from ._util import NapariMenu

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

        ACTIONS = [
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
        ]
