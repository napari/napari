from app_model import Action
from app_model.types import StandardKeyBinding, SubmenuItem, ToggleRule

from napari._app_model.actions._toggle_action import ViewerModelToggleAction
from napari._app_model.constants import MenuGroup, MenuId
from napari.components import ViewerModel
from napari.settings import get_settings


VIEW_SUBMENUS = [
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title='Axes'),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title='Scale Bar'),
    ),
]


toggle_action_details = [
    (
        'napari.window.view.toggle_viewer_axes',
        'Axes Visible',
        'axes',
        'visible',
    ),
    (
        'napari.window.view.toggle_viewer_axes_colored',
        'Axes Colored',
        'axes',
        'colored',
    ),
    (
        'napari.window.view.toggle_viewer_axes_labels',
        'Axes Labels',
        'axes',
        'labels',
    ),
    (
        'napari.window.view.toggle_viewer_axes_dashed',
        'Axes Dashed',
        'axes',
        'dashed',
    ),
    (
        'napari.window.view.toggle_viewer_axes_arrows',
        'Axes Arrows',
        'axes',
        'arrows',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar',
        'Scale Bar Visible',
        'scale_bar',
        'visible',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_box',
        'Scale Bar Box',
        'scale_bar',
        'box',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_colored',
        'Scale Bar Colored',
        'scale_bar',
        'colored',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_ticks',
        'Scale Bar Ticks',
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


VIEW_ACTIONS: list[Action] = [
    Action(
        id='napari.viewer.fit_to_view',
        title='Fit to View',
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
        title='Zoom In',
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
        title='Zoom Out',
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
        title='Toggle 2D/3D Camera',
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
        title='Toggle Layer Tooltips',
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
