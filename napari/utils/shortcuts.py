from app_model.types import KeyBinding, KeyCode, KeyMod

default_shortcuts = {
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
    'napari:activate_label_erase_mode': [KeyCode.Digit1],
    'napari:activate_fill_mode': [KeyCode.Digit3],
    'napari:activate_paint_mode': [KeyCode.Digit2],
    'napari:activate_label_pan_zoom_mode': [KeyCode.Digit5],
    'napari:activate_label_picker_mode': [KeyCode.Digit4],
    'napari:new_label': [KeyCode.KeyM],
    'napari:decrease_label_id': [KeyCode.Minus],
    'napari:increase_label_id': [KeyCode.Equal],
    'napari:decrease_brush_size': [KeyCode.BracketLeft],
    'napari:increase_brush_size': [KeyCode.BracketRight],
    'napari:toggle_preserve_labels': [KeyCode.KeyP],
    'napari:activate_points_add_mode': [KeyCode.Digit2],
    'napari:activate_points_select_mode': [KeyCode.Digit3],
    'napari:activate_points_pan_zoom_mode': [KeyCode.Digit4],
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
    'napari:activate_add_rectangle_mode': [KeyCode.KeyR],
    'napari:activate_add_ellipse_mode': [KeyCode.KeyE],
    'napari:activate_add_line_mode': [KeyCode.KeyL],
    'napari:activate_add_path_mode': [KeyCode.KeyT],
    'napari:activate_add_polygon_mode': [KeyCode.KeyP],
    'napari:activate_direct_mode': [KeyCode.Digit4],
    'napari:activate_select_mode': [KeyCode.Digit5],
    'napari:activate_shape_pan_zoom_mode': [KeyCode.Digit6],
    'napari:activate_vertex_insert_mode': [KeyCode.Digit2],
    'napari:activate_vertex_remove_mode': [KeyCode.Digit1],
    'napari:copy_selected_shapes': [KeyMod.CtrlCmd | KeyCode.KeyC],
    'napari:paste_shape': [KeyMod.CtrlCmd | KeyCode.KeyV],
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
    'napari:transform_active_layer': [
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyA
    ],
}

default_shortcuts = {
    name: [KeyBinding.from_int(kb) for kb in value]
    for name, value in default_shortcuts.items()
}
