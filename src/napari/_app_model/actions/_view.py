from app_model import Action
from app_model.types import (
    KeyBindingRule,
    StandardKeyBinding,
    SubmenuItem,
    ToggleRule,
)

from napari._app_model.actions._toggle_action import ViewerModelToggleAction
from napari._app_model.constants import MenuGroup, MenuId
from napari.components import ViewerModel
from napari.settings import get_settings

VIEW_SUBMENUS = [
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCENE_AXES, title='Scene Axes'),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_CANVAS_AXES, title='Canvas Axes'),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title='Scale Bar'),
    ),
]


toggle_actions = {
    MenuId.VIEW_SCENE_AXES: [
        (
            'napari.scene.toggle_axes',
            'Toggle Scene Axes',
            'scene.overlays.axes.visible',
        ),
        (
            'napari.scene.toggle_axes_colored',
            'Toggle Scene Axes Colored',
            'scene.overlays.axes.colored',
        ),
        (
            'napari.scene.toggle_axes_labels',
            'Toggle Scene Axes Labels',
            'scene.overlays.axes.labels',
        ),
        (
            'napari.scene.toggle_axes_dashed',
            'Toggle Scene Axes Dashed',
            'scene.overlays.axes.dashed',
        ),
        (
            'napari.scene.toggle_axes_arrows',
            'Toggle Scene Axes Arrows',
            'scene.overlays.axes.arrows',
        ),
    ],
    MenuId.VIEW_CANVAS_AXES: [
        (
            'napari.canvas.toggle_axes',
            'Toggle Canvas Axes',
            'canvas.overlays.axes.visible',
        ),
        (
            'napari.canvas.toggle_axes_box',
            'Toggle Canvas Axes Box',
            'canvas.overlays.axes.box',
        ),
        (
            'napari.canvas.toggle_axes_colored',
            'Toggle Canvas Axes Colored',
            'canvas.overlays.axes.colored',
        ),
        (
            'napari.canvas.toggle_axes_labels',
            'Toggle Canvas Axes Labels',
            'canvas.overlays.axes.labels',
        ),
        (
            'napari.canvas.toggle_axes_dashed',
            'Toggle Canvas Axes Dashed',
            'canvas.overlays.axes.dashed',
        ),
        (
            'napari.canvas.toggle_axes_arrows',
            'Toggle Canvas Axes Arrows',
            'canvas.overlays.axes.arrows',
        ),
    ],
    MenuId.VIEW_SCALEBAR: [
        (
            'napari.canvas.toggle_scale_bar',
            'Toggle Scale Bar',
            'canvas.overlays.scale_bar.visible',
        ),
        (
            'napari.canvas.toggle_scale_bar_box',
            'Toggle Scale Bar Box',
            'canvas.overlays.scale_bar.box',
        ),
        (
            'napari.canvas.toggle_scale_bar_colored',
            'Toggle Scale Bar Colored',
            'canvas.overlays.scale_bar.colored',
        ),
        (
            'napari.canvas.toggle_scale_bar_ticks',
            'Toggle Scale Bar Ticks',
            'canvas.overlays.scale_bar.ticks',
        ),
    ],
}


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


def _fit_to_view(viewer: ViewerModel) -> None:
    viewer.fit_to_view()


def _zoom_in(viewer: ViewerModel) -> None:
    viewer.scene.camera.zoom *= 1.5


def _zoom_out(viewer: ViewerModel) -> None:
    viewer.scene.camera.zoom /= 1.5


def _toggle_canvas_ndim(viewer: ViewerModel) -> None:
    """Toggle the current canvas between 3D and 2D."""
    if viewer.dims.ndisplay == 2:
        viewer.dims.ndisplay = 3
    else:  # == 3
        viewer.dims.ndisplay = 2


def _toggle_synced_camera(viewer: ViewerModel) -> None:
    """Toggle the camera synced mode between synced and separate."""
    viewer.scene.camera.synced = not viewer.scene.camera.synced


def _get_current_synced_camera(viewer: ViewerModel) -> bool:
    """Return the current synced state of the camera."""
    return viewer.scene.camera.synced


VIEW_ACTIONS: list[Action] = [
    Action(
        id='napari.scene.fit_to_view',
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
        id='napari.scene.zoom_in',
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
        id='napari.scene.zoom_out',
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
        id='napari.scene.toggle_ndisplay',
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
        id='napari.scene.toggle_synced_camera',
        title='Toggle Synced scene.overlays.axes.Camera',
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 2,
            }
        ],
        callback=_toggle_synced_camera,
        toggled=ToggleRule(get_current=_get_current_synced_camera),
        keybindings=[
            KeyBindingRule(primary='Ctrl+U', mac='Cmd+U'),
        ],
    ),
    # TODO: DOES THIS TOOLTIP THING EVEN WORK????
    Action(
        id='napari.window.toggle_layer_tooltips',
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

for menu_id, action_details in toggle_actions.items():
    for cmd, cmd_title, attribute_path in action_details:
        VIEW_ACTIONS.append(
            ViewerModelToggleAction(
                id=cmd,
                title=cmd_title,
                attribute_path=attribute_path,
                menus=[{'id': menu_id}],
            )
        )
