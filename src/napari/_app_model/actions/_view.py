from napari._app_model.actions._toggle_action import ViewerModelToggleAction
from napari._app_model.constants import MenuId
from napari.utils.translations import trans

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

VIEW_ACTIONS = []
MENUID_DICT = {'axes': MenuId.VIEW_AXES, 'scale_bar': MenuId.VIEW_SCALEBAR}


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
