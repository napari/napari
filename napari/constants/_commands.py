"""All commands that are available in the napari GUI are defined here.

Internally, prefer using the CommandId enum instead of the string literal.
When adding a new command, add a new title/description in the _COMMAND_INFO dict
below.  The title will be used in the GUI, and the may be used in auto generated
documentation.

CommandId values should be namespaced, e.g. 'napari:layer:something' for a command
that operates on layers.
"""
import sys
from typing import NamedTuple, Optional

from napari.utils.translations import trans

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # in 3.11+, using str as a mixin with enum causes it to put the enum name in f-strings
    from enum import Enum

    class StrEnum(str, Enum):
        pass


# fmt: off
class CommandId(StrEnum):
    """Id representing a napari command."""

    # File menubar
    DLG_OPEN_FILES = 'napari:window:file:open_files_dialog'
    DLG_OPEN_FILES_AS_STACK = 'napari:window:file:open_files_as_stack_dialog'
    DLG_OPEN_FOLDER = 'napari:window:file:open_folder_dialog'
    DLG_OPEN_FILES_WITH_PLUGIN = 'napari:window:file:_open_files_with_plugin'
    DLG_OPEN_FILES_AS_STACK_WITH_PLUGIN = 'napari:window:file:_open_files_as_stack_with_plugin'
    DLG_OPEN_FOLDER_WITH_PLUGIN = 'napari:window:file:_open_folder_with_plugin'
    DLG_SHOW_PREFERENCES = 'napari:window:file:show_preferences_dialog'
    DLG_SAVE_LAYERS = 'napari:window:file:save_layers_dialog'
    # `DLG_SAVE_SELECTED_LAYERS` uses the same callback as `DLG_SAVE_LAYERS`,
    # just with different kwarg
    DLG_SAVE_SELECTED_LAYERS = 'napari:window:file:save_layers_dialog:selected'
    DLG_SAVE_CANVAS_SCREENSHOT = 'napari:window:file:save_canvas_screenshot_dialog'
    DLG_SAVE_VIEWER_SCREENSHOT = 'napari:window:file:save_viewer_screenshot_dialog'
    COPY_CANVAS_SCREENSHOT = 'napari:window:file:copy_canvas_screenshot'
    COPY_VIEWER_SCREENSHOT = 'napari:window:file:copy_viewer_screenshot'
    DLG_CLOSE = 'napari:window:file:close_dialog'
    DLG_QUIT = 'napari:window:file:quit_dialog'
    RESTART = 'napari:window:file:restart'

    # View menubar
    TOGGLE_FULLSCREEN = 'napari:window:view:toggle_fullscreen'
    TOGGLE_MENUBAR = 'napari:window:view:toggle_menubar'
    TOGGLE_PLAY = 'napari:window:view:toggle_play'
    TOGGLE_LAYER_TOOLTIPS = 'napari:window:view:toggle_layer_tooltips'
    TOGGLE_ACTIVITY_DOCK = 'napari:window:view:toggle_activity_dock'

    TOGGLE_VIEWER_AXES = 'napari:window:view:toggle_viewer_axes'
    TOGGLE_VIEWER_AXES_COLORED = 'napari:window:view:toggle_viewer_axes_colored'
    TOGGLE_VIEWER_AXES_LABELS = 'napari:window:view:toggle_viewer_axes_labels'
    TOGGLE_VIEWER_AXES_DASHED = 'napari:window:view:toggle_viewer_axesdashed'
    TOGGLE_VIEWER_AXES_ARROWS = 'napari:window:view:toggle_viewer_axes_arrows'
    TOGGLE_VIEWER_SCALE_BAR = 'napari:window:view:toggle_viewer_scale_bar'
    TOGGLE_VIEWER_SCALE_BAR_COLORED = 'napari:window:view:toggle_viewer_scale_bar_colored'
    TOGGLE_VIEWER_SCALE_BAR_TICKS = 'napari:window:view:toggle_viewer_scale_bar_ticks'
    TOGGLE_VIEWER_NDISPLAY = "napari:window:view:toggle_ndisplay"

    # Help menubar
    NAPARI_GETTING_STARTED = 'napari:window:help:getting_started'
    NAPARI_TUTORIALS = 'napari:window:help:tutorials'
    NAPARI_LAYERS_GUIDE = 'napari:window:help:layers_guide'
    NAPARI_EXAMPLES = 'napari:window:help:examples'
    NAPARI_RELEASE_NOTES = 'napari:window:help:release_notes'
    NAPARI_HOMEPAGE = 'napari:window:help:homepage'
    NAPARI_INFO = 'napari:window:help:info'
    NAPARI_GITHUB_ISSUE = 'napari:window:help:github_issue'
    TOGGLE_BUG_REPORT_OPT_IN = 'napari:window:help:bug_report_opt_in'

    NAPARI_SHOW_SHORTCUTS = "napari:window:show_shortcuts"

    # Layer menubar
    LAYER_DUPLICATE = 'napari:layer:duplicate'
    LAYER_SPLIT_STACK = 'napari:layer:split_stack'
    LAYER_SPLIT_RGB = 'napari:layer:split_rgb'
    LAYER_MERGE_STACK = 'napari:layer:merge_stack'
    LAYER_TOGGLE_VISIBILITY = 'napari:layer:toggle_visibility'

    LAYER_LINK_SELECTED = 'napari:layer:link_selected_layers'
    LAYER_UNLINK_SELECTED = 'napari:layer:unlink_selected_layers'
    LAYER_SELECT_LINKED = 'napari:layer:select_linked_layers'

    LAYER_CONVERT_TO_LABELS = 'napari:layer:convert_to_labels'
    LAYER_CONVERT_TO_IMAGE = 'napari:layer:convert_to_image'

    LAYER_CONVERT_TO_INT8 = 'napari:layer:convert_to_int8'
    LAYER_CONVERT_TO_INT16 = 'napari:layer:convert_to_int16'
    LAYER_CONVERT_TO_INT32 = 'napari:layer:convert_to_int32'
    LAYER_CONVERT_TO_INT64 = 'napari:layer:convert_to_int64'
    LAYER_CONVERT_TO_UINT8 = 'napari:layer:convert_to_uint8'
    LAYER_CONVERT_TO_UINT16 = 'napari:layer:convert_to_uint16'
    LAYER_CONVERT_TO_UINT32 = 'napari:layer:convert_to_uint32'
    LAYER_CONVERT_TO_UINT64 = 'napari:layer:convert_to_uint64'

    LAYER_PROJECT_MAX = 'napari:layer:project_max'
    LAYER_PROJECT_MIN = 'napari:layer:project_min'
    LAYER_PROJECT_STD = 'napari:layer:project_std'
    LAYER_PROJECT_SUM = 'napari:layer:project_sum'
    LAYER_PROJECT_MEAN = 'napari:layer:project_mean'
    LAYER_PROJECT_MEDIAN = 'napari:layer:project_median'

    # Viewer Actions
    VIEWER_DELETE_SELECTED = "napari:viewer:delete_selected_layers"
    VIEWER_HOLD_FOR_PAN_ZOOM = "napari:viewer:hold_for_pan_zoom"
    VIEWER_RESET_SCROLL = "napari:viewer:reset_scroll_progress"
    VIEWER_TOGGLE_THEME = "napari:viewer:toggle_theme"
    VIEWER_RESET_VIEW = "napari:viewer:reset_view"
    VIEWER_INC_DIMS_LEFT = "napari:viewer:increment_dims_left"
    VIEWER_INC_DIMS_RIGHT = "napari:viewer:increment_dims_right"
    VIEWER_FOCUS_AXES_UP = "napari:viewer:focus_axes_up"
    VIEWER_FOCUS_AXES_DOWN = "napari:viewer:focus_axes_down"
    VIEWER_ROLL_AXES = "napari:viewer:roll_axes"
    VIEWER_TRANSPOSE_AXES = "napari:viewer:transpose_axes"
    VIEWER_TOGGLE_GRID = "napari:viewer:toggle_grid"
    # TODO: captured above in LAYER_TOGGLE_VISIBILITY?
    VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY = "napari:viewer:toggle_selected_visibility"
    # TODO: console action should be provided by plugin?
    VIEWER_TOGGLE_CONSOLE_VISIBILITY = "napari:viewer:toggle_console_visibility"
    VIEWER_NEW_LABELS = "napari:viewer:new_labels"
    VIEWER_NEW_SHAPES = "napari:viewer:new_shapes"
    VIEWER_NEW_POINTS = "napari:viewer:new_points"

    # Image layer actions
    IMAGE_ORIENT_PLANE_NORMAL_ALONG_Z = "napari:image:orient_plane_normal_along_z"
    IMAGE_ORIENT_PLANE_NORMAL_ALONG_Y = "napari:image:orient_plane_normal_along_y"
    IMAGE_ORIENT_PLANE_NORMAL_ALONG_X = "napari:image:orient_plane_normal_along_x"
    IMAGE_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION = "napari:image:orient_plane_normal_along_view_direction"
    IMAGE_HOLD_TO_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION = "napari:image:hold_to_orient_plane_normal_along_view_direction"
    IMAGE_ACTIVATE_TRANSFORM_MODE = "napari:image:activate_image_transform_mode"
    IMAGE_ACTIVATE_PAN_ZOOM_MODE = "napari:image:activate_image_pan_zoom_mode"

    LABELS_ACTIVATE_PAINT_MODE = "napari:labels:activate_labels_paint_mode"
    LABELS_ACTIVATE_FILL_MODE = "napari:labels:activate_labels_fill_mode"
    LABELS_ACTIVATE_PAN_ZOOM_MODE = "napari:labels:activate_labels_pan_zoom_mode"
    LABELS_ACTIVATE_PICKER_MODE = "napari:labels:activate_labels_picker_mode"
    LABELS_ACTIVATE_POLYGON_MODE = "napari:labels:activate_labels_polygon_mode"
    LABELS_ACTIVATE_ERASE_MODE = "napari:labels:activate_labels_erase_mode"
    LABELS_ACTIVATE_TRANSFORM_MODE = "napari:labels:activate_labels_transform_mode"
    LABELS_RESET_POLYGON = "napari:labels:reset_polygon"
    LABELS_COMPLETE_POLYGON = "napari:labels:complete_polygon"
    LABELS_NEW_LABEL = "napari:labels:new_label"
    LABELS_DECREMENT_ID = "napari:labels:decrease_label_id"
    LABELS_INCREMENT_ID = "napari:labels:increase_label_id"
    LABELS_DECREASE_BRUSH_SIZE = "napari:labels:decrease_brush_size"
    LABELS_INCREASE_BRUSH_SIZE = "napari:labels:increase_brush_size"
    LABELS_SWAP = "napari:labels:swap_selected_and_background_labels"
    LABELS_TOGGLE_PRESERVE_LABELS = "napari:labels:toggle_preserve_labels"
    LABELS_UNDO = "napari:labels:labels_undo"
    LABELS_REDO = "napari:labels:labels_redo"

    POINTS_ACTIVATE_ADD_MODE = "napari:points:activate_points_add_mode"
    POINTS_ACTIVATE_SELECT_MODE = "napari:points:activate_points_select_mode"
    POINTS_ACTIVATE_PAN_ZOOM_MODE = "napari:points:activate_points_pan_zoom_mode"
    POINTS_ACTIVATE_TRANSFORM_MODE = "napari:points:activate_points_transform_mode"
    POINTS_COPY = "napari:points:copy_selected_points"
    POINTS_PASTE = "napari:points:paste_points"
    POINTS_SELECT_ALL_IN_SLICE = "napari:points:select_all_in_slice"
    POINTS_SELECT_ALL_DATA = "napari:points:select_all_data"
    POINTS_DELETE_SELECTED = "napari:points:delete_selected_points"

    SHAPES_HOLD_TO_LOCK_ASPECT_RATIO = "napari:shapes:hold_to_lock_aspect_ratio"
    SHAPES_ACTIVATE_ADD_RECTANGLE_MODE = "napari:shapes:activate_add_rectangle_mode"
    SHAPES_ACTIVATE_ADD_ELLIPSE_MODE = "napari:shapes:activate_add_ellipse_mode"
    SHAPES_ACTIVATE_ADD_LINE_MODE = "napari:shapes:activate_add_line_mode"
    SHAPES_ACTIVATE_ADD_PATH_MODE = "napari:shapes:activate_add_path_mode"
    SHAPES_ACTIVATE_ADD_POLYGON_MODE = "napari:shapes:activate_add_polygon_mode"
    SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE = "napari:shapes:activate_add_polygon_lasso_mode"
    SHAPES_ACTIVATE_DIRECT_MODE = "napari:shapes:activate_direct_mode"
    SHAPES_ACTIVATE_SELECT_MODE = "napari:shapes:activate_select_mode"
    SHAPES_ACTIVATE_PAN_ZOOM_MODE = "napari:shapes:activate_shapes_pan_zoom_mode"
    SHAPES_ACTIVATE_TRANSFORM_MODE = "napari:shapes:activate_shapes_transform_mode"
    SHAPES_ACTIVATE_VERTEX_INSERT_MODE = "napari:shapes:activate_vertex_insert_mode"
    SHAPES_ACTIVATE_VERTEX_REMOVE_MODE = "napari:shapes:activate_vertex_remove_mode"
    SHAPES_COPY = "napari:shapes:copy_selected_shapes"
    SHAPES_PASTE = "napari:shapes:paste_shapes"
    SHAPES_SELECT_ALL = "napari:shapes:select_all_shapes"
    SHAPES_DELETE = "napari:shapes:delete_selected_shapes"
    SHAPES_MOVE_TO_FRONT = "napari:shapes:move_shapes_selection_to_front"
    SHAPES_MOVE_TO_BACK = "napari:shapes:move_shapes_selection_to_back"
    SHAPES_FINISH_DRAWING_SHAPE = "napari:shapes:finish_drawing_shape"

    SURFACE_ACTIVATE_PAN_ZOOM_MODE = "napari:surface:activate_surface_pan_zoom_mode"
    SURFACE_ACTIVATE_TRANSFORM_MODE = "napari:surface:activate_surface_transform_mode"

    TRACKS_ACTIVATE_PAN_ZOOM_MODE = "napari:tracks:activate_tracks_pan_zoom_mode"
    TRACKS_ACTIVATE_TRANSFORM_MODE = "napari:tracks:activate_tracks_transform_mode"

    VECTORS_ACTIVATE_PAN_ZOOM_MODE = "napari:vectors:activate_vectors_pan_zoom_mode"
    VECTORS_ACTIVATE_TRANSFORM_MODE = "napari:vectors:activate_vectors_transform_mode"

    @property
    def title(self) -> str:  # type: ignore[override]
        return _COMMAND_INFO[self].title

    @property
    def description(self) -> Optional[str]:
        return _COMMAND_INFO[self].description


class _i(NamedTuple):
    """simple utility tuple for defining items in _COMMAND_INFO."""

    title: str
    description: Optional[str] = None


_COMMAND_INFO = {
    # File menubar
    CommandId.DLG_OPEN_FILES: _i(trans._('Open File(s)...')),
    CommandId.DLG_OPEN_FILES_AS_STACK: _i(trans._('Open Files as Stack...')),
    CommandId.DLG_OPEN_FOLDER: _i(trans._('Open Folder...')),
    CommandId.DLG_OPEN_FILES_WITH_PLUGIN: _i(trans._('Open File(s)...')),
    CommandId.DLG_OPEN_FILES_AS_STACK_WITH_PLUGIN: _i(trans._('Open Files as Stack...')),
    CommandId.DLG_OPEN_FOLDER_WITH_PLUGIN: _i(trans._('Open Folder...')),
    CommandId.DLG_SHOW_PREFERENCES: _i(trans._('Preferences')),
    CommandId.DLG_SAVE_LAYERS: _i(trans._('Save All Layers...')),
    CommandId.DLG_SAVE_SELECTED_LAYERS: _i(trans._('Save Selected Layers...')),
    CommandId.DLG_SAVE_CANVAS_SCREENSHOT: _i(trans._('Save Screenshot...')),
    CommandId.DLG_SAVE_VIEWER_SCREENSHOT: _i(trans._('Save Screenshot with Viewer...')),
    CommandId.COPY_CANVAS_SCREENSHOT: _i(trans._('Copy Screenshot to Clipboard')),
    CommandId.COPY_VIEWER_SCREENSHOT: _i(trans._('Copy Screenshot with Viewer to Clipboard')),
    CommandId.DLG_CLOSE: _i(trans._('Close Window')),
    CommandId.DLG_QUIT: _i(trans._('Exit')),
    CommandId.RESTART: _i(trans._('Restart')),

    # View menubar
    CommandId.TOGGLE_FULLSCREEN: _i(trans._('Toggle Full Screen')),
    CommandId.TOGGLE_MENUBAR: _i(trans._('Toggle Menubar Visibility')),
    CommandId.TOGGLE_PLAY: _i(trans._('Toggle Play')),
    CommandId.TOGGLE_LAYER_TOOLTIPS: _i(trans._('Toggle Layer Tooltips')),
    CommandId.TOGGLE_ACTIVITY_DOCK: _i(trans._('Toggle Activity Dock')),
    CommandId.TOGGLE_VIEWER_AXES: _i(trans._('Axes Visible')),
    CommandId.TOGGLE_VIEWER_AXES_COLORED: _i(trans._('Axes Colored')),
    CommandId.TOGGLE_VIEWER_AXES_LABELS: _i(trans._('Axes Labels')),
    CommandId.TOGGLE_VIEWER_AXES_DASHED: _i(trans._('Axes Dashed')),
    CommandId.TOGGLE_VIEWER_AXES_ARROWS: _i(trans._('Axes Arrows')),
    CommandId.TOGGLE_VIEWER_SCALE_BAR: _i(trans._('Scale Bar Visible')),
    CommandId.TOGGLE_VIEWER_SCALE_BAR_COLORED: _i(trans._('Scale Bar Colored')),
    CommandId.TOGGLE_VIEWER_SCALE_BAR_TICKS: _i(trans._('Scale Bar Ticks')),
    CommandId.TOGGLE_VIEWER_NDISPLAY : _i(trans._('Toggle nDisplay')),

    # Help menubar
    CommandId.NAPARI_GETTING_STARTED: _i(trans._('Getting started')),
    CommandId.NAPARI_TUTORIALS: _i(trans._('Tutorials')),
    CommandId.NAPARI_LAYERS_GUIDE: _i(trans._('Using Layers Guides')),
    CommandId.NAPARI_EXAMPLES: _i(trans._('Examples Gallery')),
    CommandId.NAPARI_RELEASE_NOTES: _i(trans._('Release Notes')),
    CommandId.NAPARI_HOMEPAGE: _i(trans._('napari homepage')),
    CommandId.NAPARI_INFO: _i(trans._('napari Info')),
    CommandId.NAPARI_GITHUB_ISSUE: _i(trans._('Report an issue on GitHub')),
    CommandId.TOGGLE_BUG_REPORT_OPT_IN: _i(trans._('Bug Reporting Opt In/Out...')),

    CommandId.NAPARI_SHOW_SHORTCUTS: _i(trans._('Show all key bindings')),

    # Layer menubar
    CommandId.LAYER_DUPLICATE: _i(trans._('Duplicate Layer')),
    CommandId.LAYER_SPLIT_STACK: _i(trans._('Split Stack')),
    CommandId.LAYER_SPLIT_RGB: _i(trans._('Split RGB')),
    CommandId.LAYER_MERGE_STACK: _i(trans._('Merge to Stack')),
    CommandId.LAYER_TOGGLE_VISIBILITY: _i(trans._('Toggle visibility')),
    CommandId.LAYER_LINK_SELECTED: _i(trans._('Link Layers')),
    CommandId.LAYER_UNLINK_SELECTED: _i(trans._('Unlink Layers')),
    CommandId.LAYER_SELECT_LINKED: _i(trans._('Select Linked Layers')),
    CommandId.LAYER_CONVERT_TO_LABELS: _i(trans._('Convert to Labels')),
    CommandId.LAYER_CONVERT_TO_IMAGE: _i(trans._('Convert to Image')),
    CommandId.LAYER_CONVERT_TO_INT8: _i(trans._('Convert to int8')),
    CommandId.LAYER_CONVERT_TO_INT16: _i(trans._('Convert to int16')),
    CommandId.LAYER_CONVERT_TO_INT32: _i(trans._('Convert to int32')),
    CommandId.LAYER_CONVERT_TO_INT64: _i(trans._('Convert to int64')),
    CommandId.LAYER_CONVERT_TO_UINT8: _i(trans._('Convert to uint8')),
    CommandId.LAYER_CONVERT_TO_UINT16: _i(trans._('Convert to uint16')),
    CommandId.LAYER_CONVERT_TO_UINT32: _i(trans._('Convert to uint32')),
    CommandId.LAYER_CONVERT_TO_UINT64: _i(trans._('Convert to uint64')),
    CommandId.LAYER_PROJECT_MAX: _i(trans._('Max projection')),
    CommandId.LAYER_PROJECT_MIN: _i(trans._('Min projection')),
    CommandId.LAYER_PROJECT_STD: _i(trans._('Std projection')),
    CommandId.LAYER_PROJECT_SUM: _i(trans._('Sum projection')),
    CommandId.LAYER_PROJECT_MEAN: _i(trans._('Mean projection')),
    CommandId.LAYER_PROJECT_MEDIAN: _i(trans._('Median projection')),

    CommandId.VIEWER_DELETE_SELECTED: _i(trans._('Delete layers'), trans._('delete all selected layers')),
    CommandId.VIEWER_HOLD_FOR_PAN_ZOOM: _i(trans._('Hold for pan/zoom'), trans._('Press and hold for pan/zoom mode')),
    CommandId.VIEWER_RESET_SCROLL: _i(trans._('Reset scroll'), trans._('Reset dims scroll progress')),
    CommandId.VIEWER_TOGGLE_THEME: _i(trans._('Toggle theme'), trans._('Toggle current viewer theme')),
    CommandId.VIEWER_RESET_VIEW: _i(trans._('Reset view'), trans._('Reset view to original state.')),
    CommandId.VIEWER_INC_DIMS_LEFT: _i(trans._('Increment dims left')),
    CommandId.VIEWER_INC_DIMS_RIGHT: _i(trans._('Increment dims right')),
    CommandId.VIEWER_FOCUS_AXES_UP: _i(trans._('Focus axes up'), trans._('Move focus of dimensions slider up.')),
    CommandId.VIEWER_FOCUS_AXES_DOWN: _i(trans._('Focus axes down'), trans._('Move focus of dimensions slider down.')),
    # Use non-breaking spaces and non-breaking hyphen for Preferences table
    CommandId.VIEWER_ROLL_AXES: _i(trans._('Roll axes'), trans._('Change order of the visible axes, e.g.\u00A0[0,\u00A01,\u00A02]\u00A0\u2011>\u00A0[2,\u00A00,\u00A01].')),
    CommandId.VIEWER_TRANSPOSE_AXES: _i(trans._('Transpose axes'), trans._('Transpose order of the last two visible axes, e.g.\u00A0[0,\u00A01]\u00A0\u2011>\u00A0[1,\u00A00].')),
    # TODO: move to view menu?
    CommandId.VIEWER_TOGGLE_GRID: _i(trans._('Grid mode')),
    # TODO: captured above in LAYER_TOGGLE_VISIBILITY?
    CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY: _i(trans._('Toggle selected layer visibility')),
    CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY: _i(trans._('Toggle console'), trans._('Show/Hide IPython console (only available when napari started as standalone application)')),
    CommandId.VIEWER_NEW_LABELS: _i(trans._('New labels layer')),
    CommandId.VIEWER_NEW_SHAPES: _i(trans._('New shapes layer')),
    CommandId.VIEWER_NEW_POINTS: _i(trans._('New points layer')),

    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Z: _i(trans._('Orient along Z'), trans._('orient plane normal along z-axis')),
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Y: _i(trans._('Orient along Y'), trans._('orient plane normal along y-axis')),
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_X: _i(trans._('Orient along X'), trans._('orient plane normal along x-axis')),
    CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION: _i(trans._('Orient along view'), trans._('orient plane normal along view axis')),
    CommandId.IMAGE_HOLD_TO_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION: _i(trans._('Orient along view'), trans._('hold to orient plane normal along view axis')),
    CommandId.IMAGE_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate tranform mode')),
    CommandId.IMAGE_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),

    CommandId.POINTS_ACTIVATE_ADD_MODE: _i(trans._('Add points'), trans._('activate the adder tool')),
    CommandId.POINTS_ACTIVATE_SELECT_MODE: _i(trans._('Select points'), trans._('activate the selection tool')),
    CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),
    CommandId.POINTS_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate the transform tool')),
    CommandId.POINTS_COPY: _i(trans._('Copy'), trans._('copy any selected points')),
    CommandId.POINTS_PASTE: _i(trans._('Paste'), trans._('paste any copied points')),
    CommandId.POINTS_SELECT_ALL_IN_SLICE: _i(trans._('Select all in current slice'), trans._('select all points in the current view slice')),
    CommandId.POINTS_SELECT_ALL_DATA: _i(trans._('Select all in layer'), trans._('select all points in the layer')),
    CommandId.POINTS_DELETE_SELECTED: _i(trans._('Delete points'), trans._('delete all selected points')),

    CommandId.LABELS_ACTIVATE_PAINT_MODE: _i(trans._('Paint'), trans._('activate the paint brush tool')),
    CommandId.LABELS_ACTIVATE_FILL_MODE: _i(trans._('Fill'), trans._('activate the fill bucket tool')),
    CommandId.LABELS_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),
    CommandId.LABELS_ACTIVATE_PICKER_MODE: _i(trans._('Color Picker'), trans._('activate the color picker tool')),
    CommandId.LABELS_ACTIVATE_POLYGON_MODE: _i(trans._('Draw Polygon'), trans._('activate polygon tool')),
    CommandId.LABELS_ACTIVATE_ERASE_MODE: _i(trans._('Erase'), trans._('activate the label eraser')),
    CommandId.LABELS_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate the transform tool')),
    CommandId.LABELS_RESET_POLYGON: _i(trans._('Reset polygon'), trans._('Reset the drawing of the current polygon')),
    CommandId.LABELS_COMPLETE_POLYGON: _i(trans._('Complete polygon'), trans._('Complete the current polygon')),
    CommandId.LABELS_NEW_LABEL: _i(trans._('New label'), trans._('set the currently selected label to the largest used label plus one')),
    CommandId.LABELS_DECREMENT_ID: _i(trans._('Decrement label'), trans._('decrease the currently selected label by one')),
    CommandId.LABELS_INCREMENT_ID: _i(trans._('Increment label'), trans._('increase the currently selected label by one')),
    CommandId.LABELS_DECREASE_BRUSH_SIZE: _i(trans._('Decrease brush size')),
    CommandId.LABELS_INCREASE_BRUSH_SIZE: _i(trans._('Increase brush size')),
    CommandId.LABELS_SWAP: _i(trans._('Swap labels'), trans._('Swap between the selected label and the background label')),
    CommandId.LABELS_TOGGLE_PRESERVE_LABELS: _i(trans._('Toggle preserve labels')),
    CommandId.LABELS_UNDO: _i(trans._('Undo'), trans._('undo the last paint or fill action since the view slice has changed')),
    CommandId.LABELS_REDO: _i(trans._('Redo'), trans._('redo any previously undone actions')),

    CommandId.SHAPES_HOLD_TO_LOCK_ASPECT_RATIO: _i(trans._('Hold to lock aspect ratio'), trans._('hold to lock aspect ratio when resizing a shape')),
    CommandId.SHAPES_ACTIVATE_ADD_RECTANGLE_MODE: _i(trans._('Add rectangles'), trans._('activate the add rectangle tool')),
    CommandId.SHAPES_ACTIVATE_ADD_ELLIPSE_MODE: _i(trans._('Add ellipses'), trans._('activate the add ellipse tool')),
    CommandId.SHAPES_ACTIVATE_ADD_LINE_MODE: _i(trans._('Add lines'), trans._('activate the add line tool')),
    CommandId.SHAPES_ACTIVATE_ADD_PATH_MODE: _i(trans._('Add paths'), trans._('activate the add path tool')),
    CommandId.SHAPES_ACTIVATE_ADD_POLYGON_MODE: _i(trans._('Add polygons'), trans._('activate the add polygon tool')),
    CommandId.SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE: _i(trans._('Add polygons lasso'), trans._('activate the add polygons lasso tool')),
    CommandId.SHAPES_ACTIVATE_DIRECT_MODE: _i(trans._('Select vertices'), trans._('activate the vertex selection tool')),
    CommandId.SHAPES_ACTIVATE_SELECT_MODE: _i(trans._('Select shapes'), trans._('activate the shape selection tool')),
    CommandId.SHAPES_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),
    CommandId.SHAPES_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate the transform tool')),
    CommandId.SHAPES_ACTIVATE_VERTEX_INSERT_MODE: _i(trans._('Insert vertex'), trans._('activate the vertex insertion tool')),
    CommandId.SHAPES_ACTIVATE_VERTEX_REMOVE_MODE: _i(trans._('Remove vertex'), trans._('activate the vertex deletion tool')),
    CommandId.SHAPES_COPY: _i(trans._('Copy'), trans._('copy any selected shapes')),
    CommandId.SHAPES_PASTE: _i(trans._('Paste'), trans._('paste any copied shapes')),
    CommandId.SHAPES_SELECT_ALL: _i(trans._('Select all'), trans._('select all shapes in the current view slice')),
    CommandId.SHAPES_DELETE: _i(trans._('Delete'), trans._('delete any selected shapes')),
    CommandId.SHAPES_MOVE_TO_FRONT: _i(trans._('Move to front')),
    CommandId.SHAPES_MOVE_TO_BACK: _i(trans._('Move to back')),
    CommandId.SHAPES_FINISH_DRAWING_SHAPE: _i(trans._('Finish drawing shape'), trans._('finish any drawing, for example using the path or polygon tool')),

    CommandId.SURFACE_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),
    CommandId.SURFACE_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate the transform tool')),

    CommandId.TRACKS_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),
    CommandId.TRACKS_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate the transform tool')),

    CommandId.VECTORS_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode')),
    CommandId.VECTORS_ACTIVATE_TRANSFORM_MODE: _i(trans._('Transform'), trans._('activate the transform tool')),
}
# fmt: on
