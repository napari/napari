"""Qt 'Plugins' menu Actions."""

from logging import getLogger

from app_model.types import Action

from napari._app_model.constants import MenuId
from napari.components._viewer_key_bindings import (
    reset_view,
    roll_axes,
    toggle_console_visibility,
    toggle_grid,
    toggle_ndisplay,
    transpose_axes,
)
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
    Action(
        id='napari.viewer.toggle_ndisplay',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=toggle_ndisplay,
        tooltip=trans._('Toggle 2D/3D view.'),
        toggled=False,
    ),
    Action(
        id='napari.viewer.roll_axes',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=roll_axes,
        tooltip=trans._(
            'Change order of the visible axes, e.g.\u00a0[0,\u00a01,\u00a02]\u00a0\u2011>\u00a0[2,\u00a00,\u00a01].'
        ),
    ),
    Action(
        id='napari.viewer.transpose_axes',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=transpose_axes,
        tooltip=trans._(
            'Transpose order of the last two visible axes, e.g.\u00a0[0,\u00a01]\u00a0\u2011>\u00a0[1,\u00a00].'
        ),
    ),
    Action(
        id='napari.viewer.toggle_grid',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=toggle_grid,
        tooltip=trans._('Toggle grid mode.'),
    ),
    Action(
        id='napari.viewer.reset_view',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=reset_view,
        tooltip=trans._('Reset view to original state.'),
    ),
]
