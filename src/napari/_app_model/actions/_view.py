from app_model import Action
from app_model.types import StandardKeyBinding, SubmenuItem, ToggleRule

from napari._app_model.actions._toggle_action import ViewerModelToggleAction
from napari._app_model.constants import MenuGroup, MenuId
from napari.components import ViewerModel
from napari.settings import get_settings
from napari.utils.translations import trans

VIEW_SUBMENUS = [
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title=trans._('Axes')),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title=trans._('Scale Bar')),
    ),
]


toggle_action_details = [
    (
        'napari.window.view.toggle_viewer_axes',
        trans._('Axes Visible'),
        'axes',
        'visible',
    ),
    (
        'napari.window.view.toggle_viewer_axes_colored',
        trans._('Axes Colored'),
        'axes',
        'colored',
    ),
    (
        'napari.window.view.toggle_viewer_axes_labels',
        trans._('Axes Labels'),
        'axes',
        'labels',
    ),
    (
        'napari.window.view.toggle_viewer_axes_dashed',
        trans._('Axes Dashed'),
        'axes',
        'dashed',
    ),
    (
        'napari.window.view.toggle_viewer_axes_arrows',
        trans._('Axes Arrows'),
        'axes',
        'arrows',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar',
        trans._('Scale Bar Visible'),
        'scale_bar',
        'visible',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_box',
        trans._('Scale Bar Box'),
        'scale_bar',
        'box',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_colored',
        trans._('Scale Bar Colored'),
        'scale_bar',
        'colored',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_ticks',
        trans._('Scale Bar Ticks'),
        'scale_bar',
        'ticks',
    ),
]

MENUID_DICT = {'axes': MenuId.VIEW_AXES, 'scale_bar': MenuId.VIEW_SCALEBAR}


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


def _fit_to_view(viewer: ViewerModel) -> None:
    viewer.fit_to_view()


def _zoom_in(viewer: ViewerModel) -> None:
    viewer.camera.zoom *= 1.5


def _zoom_out(viewer: ViewerModel) -> None:
    viewer.camera.zoom /= 1.5


def _toggle_canvas_ndim(viewer: ViewerModel) -> None:
    """Toggle the current canvas between 3D and 2D."""
    if viewer.dims.ndisplay == 2:
        viewer.dims.ndisplay = 3
    else:  # == 3
        viewer.dims.ndisplay = 2


VIEW_ACTIONS = [
    Action(
        id='napari.viewer.fit_to_view',
        title=trans._('Fit to View'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_fit_to_view,
        keybindings=[StandardKeyBinding.OriginalSize],
    ),
    Action(
        id='napari.viewer.camera.zoom_in',
        title=trans._('Zoom In'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_zoom_in,
        keybindings=[StandardKeyBinding.ZoomIn],
    ),
    Action(
        id='napari.viewer.camera.zoom_out',
        title=trans._('Zoom Out'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_zoom_out,
        keybindings=[StandardKeyBinding.ZoomOut],
    ),
    # TODO: this could be made into a toggle setting Action subclass
    # using a similar pattern to the above ViewerToggleAction classes
    Action(
        id='napari.window.view.toggle_ndisplay',
        title=trans._('Toggle 2D/3D Camera'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 2,
            }
        ],
        callback=_toggle_canvas_ndim,
    ),
    Action(
        id='napari.window.view.toggle_layer_tooltips',
        title=trans._('Toggle Layer Tooltips'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.RENDER,
                'order': 10,
            }
        ],
        callback=_tooltip_visibility_toggle,
        toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
    ),
]

for cmd, cmd_title, viewer_attr, sub_attr in toggle_action_details:
    VIEW_ACTIONS.append(
        ViewerModelToggleAction(
            id=cmd,
            title=cmd_title,
            viewer_attribute=viewer_attr,
            sub_attribute=sub_attr,
            menus=[{'id': MENUID_DICT[viewer_attr]}],
        )
    )
