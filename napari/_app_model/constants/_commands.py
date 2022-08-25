"""All commands that are available in the napari GUI are defined here.

Internally, prefer using the CommandId enum instead of the string literal.
When adding a new command, add a new title/description in the _COMMAND_INFO dict
below.  The title will be used in the GUI, and the may be used in auto generated
documentation.

CommandId values should be namespaced, e.g. 'napari:layer:something' for a command
that operates on layers.
"""
from enum import Enum
from typing import NamedTuple, Optional

from ...utils.translations import trans


class CommandId(str, Enum):
    """Id representing a napari command."""

    TOGGLE_CONSOLE = "napari:window:window:toggle_window_console"
    TOGGLE_LAYER_CONTROLS = "napari:window:window:toggle_layer_controls"
    TOGGLE_LAYER_LIST = "napari:window:window:toggle_layer_list"

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

    @property
    def title(self) -> str:
        return _COMMAND_INFO[self].title

    @property
    def description(self) -> Optional[str]:
        return _COMMAND_INFO[self].description


class _i(NamedTuple):
    """simple utility tuple for defining items in _COMMAND_INFO."""

    title: str
    description: Optional[str] = None


# fmt: off
_COMMAND_INFO = {
    CommandId.TOGGLE_CONSOLE: _i(trans._('console')),
    CommandId.TOGGLE_LAYER_CONTROLS: _i(trans._('layer controls')),
    CommandId.TOGGLE_LAYER_LIST: _i(trans._('layer list')),
    CommandId.LAYER_DUPLICATE: _i(trans._('Duplicate Layer'),),
    CommandId.LAYER_SPLIT_STACK: _i(trans._('Split Stack'),),
    CommandId.LAYER_SPLIT_RGB: _i(trans._('Split RGB'),),
    CommandId.LAYER_MERGE_STACK: _i(trans._('Merge to Stack'),),
    CommandId.LAYER_TOGGLE_VISIBILITY: _i(trans._('Toggle visibility'),),
    CommandId.LAYER_LINK_SELECTED: _i(trans._('Link Layers'),),
    CommandId.LAYER_UNLINK_SELECTED: _i(trans._('Unlink Layers'),),
    CommandId.LAYER_SELECT_LINKED: _i(trans._('Select Linked Layers'),),
    CommandId.LAYER_CONVERT_TO_LABELS: _i(trans._('Convert to Labels'),),
    CommandId.LAYER_CONVERT_TO_IMAGE: _i(trans._('Convert to Image'),),
    CommandId.LAYER_CONVERT_TO_INT8: _i(trans._('Convert to int8'),),
    CommandId.LAYER_CONVERT_TO_INT16: _i(trans._('Convert to int16'),),
    CommandId.LAYER_CONVERT_TO_INT32: _i(trans._('Convert to int32'),),
    CommandId.LAYER_CONVERT_TO_INT64: _i(trans._('Convert to int64'),),
    CommandId.LAYER_CONVERT_TO_UINT8: _i(trans._('Convert to uint8'),),
    CommandId.LAYER_CONVERT_TO_UINT16: _i(trans._('Convert to uint16'),),
    CommandId.LAYER_CONVERT_TO_UINT32: _i(trans._('Convert to uint32'),),
    CommandId.LAYER_CONVERT_TO_UINT64: _i(trans._('Convert to uint64'),),
    CommandId.LAYER_PROJECT_MAX: _i(trans._('Max projection'),),
    CommandId.LAYER_PROJECT_MIN: _i(trans._('Min projection'),),
    CommandId.LAYER_PROJECT_STD: _i(trans._('Std projection'),),
    CommandId.LAYER_PROJECT_SUM: _i(trans._('Sum projection'),),
    CommandId.LAYER_PROJECT_MEAN: _i(trans._('Mean projection'),),
    CommandId.LAYER_PROJECT_MEDIAN: _i(trans._('Median projection'),),
}
# fmt: on
