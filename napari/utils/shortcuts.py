from collections import defaultdict
from typing import Dict, List

from app_model.types import KeyBinding, KeyBindingRule, KeyCode, KeyMod

default_shortcuts = {
    # viewer
    'napari:toggle_console_visibility': [
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC
    ],
    'napari:reset_scroll_progress': [KeyCode.Ctrl],
    'napari:toggle_ndisplay': [KeyMod.CtrlCmd | KeyCode.KeyY],
    'napari:toggle_theme': [KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyT],
    'napari:reset_view': [KeyMod.CtrlCmd | KeyCode.KeyR],
    'napari:show_shortcuts': [KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.Slash],
    'napari:increment_dims_left': [KeyCode.LeftArrow],
    'napari:increment_dims_right': [KeyCode.RightArrow],
    'napari:focus_axes_up': [KeyMod.Alt | KeyCode.UpArrow],
    'napari:focus_axes_down': [KeyMod.Alt | KeyCode.DownArrow],
    'napari:roll_axes': [KeyMod.CtrlCmd | KeyCode.KeyE],
    'napari:transpose_axes': [KeyMod.CtrlCmd | KeyCode.KeyT],
    'napari:toggle_grid': [KeyMod.CtrlCmd | KeyCode.KeyG],
    'napari:toggle_selected_visibility': [KeyCode.KeyG],
    # labels
    'napari:activate_labels_erase_mode': [KeyCode.Digit1],
    'napari:activate_labels_paint_mode': [KeyCode.Digit2],
    'napari:activate_labels_fill_mode': [KeyCode.Digit3],
    'napari:activate_labels_picker_mode': [KeyCode.Digit4],
    'napari:activate_labels_pan_zoom_mode': [KeyCode.Digit5],
    'napari:activate_labels_transform_mode': [KeyCode.Digit6],
    'napari:new_label': [KeyCode.KeyM],
    'napari:decrease_label_id': [KeyCode.Minus],
    'napari:increase_label_id': [KeyCode.Equal],
    'napari:decrease_brush_size': [KeyCode.BracketLeft],
    'napari:increase_brush_size': [KeyCode.BracketRight],
    'napari:toggle_preserve_labels': [KeyCode.KeyP],
    # points
    'napari:activate_points_add_mode': [KeyCode.Digit2],
    'napari:activate_points_select_mode': [KeyCode.Digit3],
    'napari:activate_points_pan_zoom_mode': [KeyCode.Digit4],
    'napari:activate_points_transform_mode': [KeyCode.Digit5],
    'napari:select_all_in_slice': [
        KeyCode.KeyA,
        KeyMod.CtrlCmd | KeyCode.KeyA,
    ],
    'napari:select_all_data': [KeyMod.Shift | KeyCode.KeyA],
    'napari:delete_selected_points': [
        KeyCode.Backspace,
        KeyCode.Delete,
        KeyCode.Digit1,
    ],
    # shapes
    'napari:activate_add_rectangle_mode': [KeyCode.KeyR],
    'napari:activate_add_ellipse_mode': [KeyCode.KeyE],
    'napari:activate_add_line_mode': [KeyCode.KeyL],
    'napari:activate_add_path_mode': [KeyCode.KeyT],
    'napari:activate_add_polygon_mode': [KeyCode.KeyP],
    'napari:activate_direct_mode': [KeyCode.Digit4],
    'napari:activate_select_mode': [KeyCode.Digit5],
    'napari:activate_shapes_pan_zoom_mode': [KeyCode.Digit6],
    'napari:activate_shapes_transform_mode': [KeyCode.Digit2],
    'napari:activate_vertex_insert_mode': [KeyCode.Digit2],
    'napari:activate_vertex_remove_mode': [KeyCode.Digit1],
    'napari:copy_selected_shapes': [KeyMod.CtrlCmd | KeyCode.KeyC],
    'napari:paste_shapes': [KeyMod.CtrlCmd | KeyCode.KeyV],
    'napari:move_shapes_selection_to_front': [KeyCode.KeyF],
    'napari:move_shapes_selection_to_back': [KeyCode.KeyB],
    'napari:select_all_shapes': [KeyCode.KeyA],
    'napari:delete_selected_shapes': [
        KeyCode.Backspace,
        KeyCode.Delete,
        KeyCode.Digit3,
    ],
    'napari:finish_drawing_shape': [KeyCode.Escape],
    'napari:reset_active_layer_affine': [
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyR
    ],
    'napari:image_hold_to_pan_zoom': [KeyCode.Space],
    'napari:labels_hold_to_pan_zoom': [KeyCode.Space],
    'napari:points_hold_to_pan_zoom': [KeyCode.Space],
    'napari:shapes_hold_to_pan_zoom': [KeyCode.Space],
    # from shapes
    'napari:hold_to_lock_aspect_ratio': [KeyCode.Shift],
    # from image
    'napari:orient_plane_normal_along_x': [KeyCode.KeyX],
    'napari:orient_plane_normal_along_y': [KeyCode.KeyY],
    'napari:orient_plane_normal_along_z': [KeyCode.KeyZ],
    'napari:orient_plane_normal_along_view_direction': [KeyCode.KeyO],
    # from points
    'napari:copy_selected_points': [KeyMod.CtrlCmd | KeyCode.KeyC],
    'napari:paste_points': [KeyMod.CtrlCmd | KeyCode.KeyV],
    # from labels
    'napari:labels_undo': [KeyMod.CtrlCmd | KeyCode.KeyZ],
    'napari:labels_redo': [KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ],
    # image
    'napari:activate_image_pan_zoom_mode': [KeyCode.Digit1],
    'napari:activate_image_transform_mode': [KeyCode.Digit2],
    # vectors
    'napari:activate_vectors_pan_zoom_mode': [KeyCode.Digit1],
    'napari:activate_vectors_transform_mode': [KeyCode.Digit2],
    # tracks
    'napari:activate_tracks_pan_zoom_mode': [KeyCode.Digit1],
    'napari:activate_tracks_transform_mode': [KeyCode.Digit2],
    # surface
    'napari:activate_surface_pan_zoom_mode': [KeyCode.Digit1],
    'napari:activate_surface_transform_mode': [KeyCode.Digit2],
}

default_shortcuts = {
    name: [KeyBinding.from_int(kb) for kb in value]
    for name, value in default_shortcuts.items()
}

plugins_shortcuts: Dict[str, List[KeyBindingRule]] = defaultdict(list)
