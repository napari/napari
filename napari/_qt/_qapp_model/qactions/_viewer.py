"""Qt Viewer Actions."""

from logging import getLogger
from typing import TYPE_CHECKING, Union

from app_model.types import Action, KeyCode, KeyMod, ToggleRule

from napari._app_model.constants import MenuId
from napari._app_model.context import ViewerContextKeys as VCK
from napari.components import _viewer_key_bindings
from napari.resources._icons import _theme_path
from napari.utils.misc import in_ipython, in_jupyter, in_python_repl
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import Viewer, ViewerModel


EMPTY_SVG_PATH = f'file:///{_theme_path("dark")}/empty.svg'
logger = getLogger(__name__)


def _get_viewer_ndisplay_status(
    viewer: Union['ViewerModel', 'Viewer'],
) -> bool:
    return viewer.dims.ndisplay == 3


def _get_viewer_grid_status(viewer: Union['ViewerModel', 'Viewer']) -> bool:
    return viewer.grid.enabled


def add_new_points(viewer: Union['ViewerModel', 'Viewer']) -> None:
    viewer.add_points(  # type: ignore[union-attr]
        ndim=max(viewer.dims.ndim, 2),
        scale=viewer.layers.extent.step,
    )


def add_new_shapes(viewer: Union['ViewerModel', 'Viewer']) -> None:
    viewer.add_shapes(  # type: ignore[union-attr]
        ndim=max(viewer.dims.ndim, 2),
        scale=viewer.layers.extent.step,
    )


def add_new_labels(viewer: Union['ViewerModel', 'Viewer']) -> None:
    viewer._new_labels()


Q_VIEWER_ACTIONS: list[Action] = [
    Action(
        id='napari.viewer.toggle_console_visibility',
        title=trans._('Toggle console visibility'),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.toggle_console_visibility,
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC}
        ],
        tooltip=trans._(
            'Show/Hide IPython console (only available when napari started as standalone application).'
        ),
        enablement=not (in_ipython() or in_jupyter() or in_python_repl()),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.toggle_ndisplay',
        title=trans._('Toggle 2D/3D view'),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.toggle_ndisplay,
        tooltip=trans._('Toggle 2D/3D view.'),
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.KeyY}],
        toggled=ToggleRule(
            condition=VCK.is_viewer_3d, get_current=_get_viewer_ndisplay_status
        ),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.roll_axes',
        title=trans._('Roll axes'),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.roll_axes,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.KeyE}],
        tooltip=trans._(
            'Change order of the visible axes, e.g.\u00a0[0,\u00a01,\u00a02]\u00a0\u2011>\u00a0[2,\u00a00,\u00a01].'
        ),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.transpose_axes',
        title=trans._('Transpose axes'),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.transpose_axes,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.KeyT}],
        tooltip=trans._(
            'Transpose order of the last two visible axes, e.g.\u00a0[0,\u00a01]\u00a0\u2011>\u00a0[1,\u00a00].'
        ),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.toggle_grid',
        title=trans._('Toggle grid mode'),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.toggle_grid,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.KeyG}],
        tooltip=trans._('Toggle grid mode.'),
        toggled=ToggleRule(
            condition=VCK.is_viewer_grid_enabled,
            get_current=_get_viewer_grid_status,
        ),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.reset_view',
        title=trans._('Reset view'),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.reset_view,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.KeyR}],
        tooltip=trans._('Reset view to original state.'),
        icon=EMPTY_SVG_PATH,
    ),
]

Q_VIEWER_NEW_DELETE_ACTIONS: list[Action] = [
    Action(
        id='napari.viewer.new_layer.new_points',
        title=trans._('New points layer'),
        menus=[{'id': MenuId.VIEWER_NEW_DELETE_LAYER}],
        callback=add_new_points,
        tooltip=trans._('New points layer.'),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.new_layer.new_shapes',
        title=trans._('New shapes layer'),
        menus=[{'id': MenuId.VIEWER_NEW_DELETE_LAYER}],
        callback=add_new_shapes,
        tooltip=trans._('New shapes layer.'),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.new_layer.new_labels',
        title=trans._('New labels layer'),
        menus=[{'id': MenuId.VIEWER_NEW_DELETE_LAYER}],
        callback=add_new_labels,
        tooltip=trans._('New labels layer.'),
        icon=EMPTY_SVG_PATH,
    ),
    Action(
        id='napari.viewer.delete_selected_layers',
        title=trans._('Delete selected layers'),
        menus=[{'id': MenuId.VIEWER_NEW_DELETE_LAYER}],
        callback=_viewer_key_bindings.delete_selected_layers,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.Delete}],
        tooltip=trans._('Delete selected layers.'),
        icon=EMPTY_SVG_PATH,
    ),
]
