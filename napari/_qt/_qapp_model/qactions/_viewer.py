"""Qt 'Plugins' menu Actions."""

from logging import getLogger

from app_model.types import Action

from napari._app_model.constants import MenuId
from napari.components._viewer_key_bindings import toggle_console_visibility
from napari.utils.translations import trans

logger = getLogger(__name__)

Q_VIEWER_ACTIONS: list[Action] = [
    Action(
        id='napari.viewer.toggle_console_visibility',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=toggle_console_visibility,
        tooltip=trans._(
            'Show/Hide IPython console (only available when napari started as standalone application)'
        ),
    ),
]
