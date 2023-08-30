from typing import Dict, List

from app_model.types import KeyBindingRule, KeyCode, KeyMod

from napari._app_model.constants import CommandId
from napari.utils.kb.constants import KeyBindingWeights

_default_shortcuts: Dict[CommandId, List[int]] = {
    # viewer
    CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY: [
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC
    ],
    CommandId.VIEWER_RESET_SCROLL: [KeyCode.Ctrl],
    CommandId.TOGGLE_VIEWER_NDISPLAY: [KeyMod.CtrlCmd | KeyCode.KeyY],
    CommandId.VIEWER_TOGGLE_THEME: [
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyT
    ],
    CommandId.VIEWER_RESET_VIEW: [KeyMod.CtrlCmd | KeyCode.KeyR],
    CommandId.VIEWER_DELETE_SELECTED: [KeyMod.CtrlCmd | KeyCode.Delete],
    CommandId.NAPARI_SHOW_SHORTCUTS: [
        KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.Slash
    ],
    CommandId.VIEWER_INC_DIMS_LEFT: [KeyCode.LeftArrow],
    CommandId.VIEWER_INC_DIMS_RIGHT: [KeyCode.RightArrow],
    CommandId.VIEWER_FOCUS_AXES_UP: [KeyMod.Alt | KeyCode.UpArrow],
    CommandId.VIEWER_FOCUS_AXES_DOWN: [KeyMod.Alt | KeyCode.DownArrow],
    CommandId.VIEWER_ROLL_AXES: [KeyMod.CtrlCmd | KeyCode.KeyE],
    CommandId.VIEWER_TRANSPOSE_AXES: [KeyMod.CtrlCmd | KeyCode.KeyT],
    CommandId.VIEWER_TOGGLE_GRID: [KeyMod.CtrlCmd | KeyCode.KeyG],
    CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY: [KeyCode.KeyG],
    CommandId.VIEWER_HOLD_FOR_PAN_ZOOM: [KeyCode.Space],
    # image
    CommandId.IMAGE_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit1],
    CommandId.IMAGE_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit2],
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_X: [KeyCode.KeyX],
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Y: [KeyCode.KeyY],
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Z: [KeyCode.KeyZ],
    CommandId.IMAGE_HOLD_TO_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION: [
        KeyCode.KeyO
    ],
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION: [
        KeyMod.CtrlCmd | KeyCode.KeyO
    ],
    # labels
    CommandId.LABELS_ACTIVATE_ERASE_MODE: [KeyCode.Digit1, KeyCode.KeyE],
    CommandId.LABELS_ACTIVATE_PAINT_MODE: [KeyCode.Digit2, KeyCode.KeyP],
    CommandId.LABELS_ACTIVATE_POLYGON_MODE: [KeyCode.Digit3],
    CommandId.LABELS_ACTIVATE_FILL_MODE: [KeyCode.Digit4, KeyCode.KeyF],
    CommandId.LABELS_ACTIVATE_PICKER_MODE: [KeyCode.Digit5, KeyCode.KeyL],
    CommandId.LABELS_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit6, KeyCode.KeyZ],
    CommandId.LABELS_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit7],
    CommandId.LABELS_NEW_LABEL: [KeyCode.KeyM],
    CommandId.LABELS_SWAP: [KeyCode.KeyX],
    CommandId.LABELS_DECREMENT_ID: [KeyCode.Minus],
    CommandId.LABELS_INCREMENT_ID: [KeyCode.Equal],
    CommandId.LABELS_DECREASE_BRUSH_SIZE: [KeyCode.BracketLeft],
    CommandId.LABELS_INCREASE_BRUSH_SIZE: [KeyCode.BracketRight],
    CommandId.LABELS_TOGGLE_PRESERVE_LABELS: [KeyCode.KeyB],
    CommandId.LABELS_RESET_POLYGON: [KeyCode.Escape],
    CommandId.LABELS_COMPLETE_POLYGON: [KeyCode.Enter],
    CommandId.LABELS_UNDO: [KeyMod.CtrlCmd | KeyCode.KeyZ],
    CommandId.LABELS_REDO: [KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ],
    # points
    CommandId.POINTS_ACTIVATE_ADD_MODE: [KeyCode.Digit2, KeyCode.KeyP],
    CommandId.POINTS_ACTIVATE_SELECT_MODE: [KeyCode.Digit3, KeyCode.KeyS],
    CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit4, KeyCode.KeyZ],
    CommandId.POINTS_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit5],
    CommandId.POINTS_SELECT_ALL_IN_SLICE: [
        KeyCode.KeyA,
        KeyMod.CtrlCmd | KeyCode.KeyA,
    ],
    CommandId.POINTS_SELECT_ALL_DATA: [KeyMod.Shift | KeyCode.KeyA],
    CommandId.POINTS_DELETE_SELECTED: [
        KeyCode.Backspace,
        KeyCode.Delete,
        KeyCode.Digit1,
    ],
    CommandId.POINTS_COPY: [KeyMod.CtrlCmd | KeyCode.KeyC],
    CommandId.POINTS_PASTE: [KeyMod.CtrlCmd | KeyCode.KeyV],
    # shapes
    CommandId.SHAPES_ACTIVATE_ADD_RECTANGLE_MODE: [KeyCode.KeyR],
    CommandId.SHAPES_ACTIVATE_ADD_ELLIPSE_MODE: [KeyCode.KeyE],
    CommandId.SHAPES_ACTIVATE_ADD_LINE_MODE: [KeyCode.KeyL],
    CommandId.SHAPES_ACTIVATE_ADD_PATH_MODE: [KeyCode.KeyT],
    CommandId.SHAPES_ACTIVATE_ADD_POLYGON_MODE: [KeyCode.KeyP],
    CommandId.SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE: [
        KeyMod.Shift | KeyCode.KeyP
    ],
    CommandId.SHAPES_ACTIVATE_DIRECT_MODE: [KeyCode.Digit4, KeyCode.KeyD],
    CommandId.SHAPES_ACTIVATE_SELECT_MODE: [KeyCode.Digit5, KeyCode.KeyS],
    CommandId.SHAPES_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit6, KeyCode.KeyZ],
    CommandId.SHAPES_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit7],
    CommandId.SHAPES_ACTIVATE_VERTEX_INSERT_MODE: [
        KeyCode.Digit2,
        KeyCode.KeyI,
    ],
    CommandId.SHAPES_ACTIVATE_VERTEX_REMOVE_MODE: [
        KeyCode.Digit1,
        KeyCode.KeyX,
    ],
    CommandId.SHAPES_COPY: [KeyMod.CtrlCmd | KeyCode.KeyC],
    CommandId.SHAPES_PASTE: [KeyMod.CtrlCmd | KeyCode.KeyV],
    CommandId.SHAPES_MOVE_TO_FRONT: [KeyCode.KeyF],
    CommandId.SHAPES_MOVE_TO_BACK: [KeyCode.KeyB],
    CommandId.SHAPES_SELECT_ALL: [KeyCode.KeyA],
    CommandId.SHAPES_DELETE: [
        KeyCode.Backspace,
        KeyCode.Delete,
        KeyCode.Digit3,
    ],
    CommandId.SHAPES_FINISH_DRAWING_SHAPE: [KeyCode.Escape],
    CommandId.SHAPES_HOLD_TO_LOCK_ASPECT_RATIO: [KeyCode.Shift],
    # vectors
    CommandId.VECTORS_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit1],
    CommandId.VECTORS_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit2],
    # tracks
    CommandId.TRACKS_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit1],
    CommandId.TRACKS_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit2],
    # surface
    CommandId.SURFACE_ACTIVATE_PAN_ZOOM_MODE: [KeyCode.Digit1],
    CommandId.SURFACE_ACTIVATE_TRANSFORM_MODE: [KeyCode.Digit2],
}

DEFAULT_SHORTCUTS: Dict[CommandId, List[KeyBindingRule]] = {
    cmd: [
        KeyBindingRule(primary=entry, weight=KeyBindingWeights.CORE)
        for entry in entries
    ]
    for cmd, entries in _default_shortcuts.items()
}
