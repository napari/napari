"""All commands that are available in the napari GUI are defined here.

"""
from enum import Enum
from typing import NamedTuple, Optional

from ...utils.translations import trans


class CommandId(str, Enum):
    LAYER_DUPLICATE = 'napari:layer:duplicate_layer'
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


class _ci(NamedTuple):
    title: str
    description: Optional[str] = None


# fmt: off
_COMMAND_INFO = {
    CommandId.LAYER_DUPLICATE: _ci(trans._('Duplicate Layer'),),
    CommandId.LAYER_SPLIT_STACK: _ci(trans._('Split Stack'),),
    CommandId.LAYER_SPLIT_RGB: _ci(trans._('Split RGB'),),
    CommandId.LAYER_MERGE_STACK: _ci(trans._('Merge to Stack'),),
    CommandId.LAYER_TOGGLE_VISIBILITY: _ci(trans._('Toggle visibility'),),
    CommandId.LAYER_LINK_SELECTED: _ci(trans._('Link Layers'),),
    CommandId.LAYER_UNLINK_SELECTED: _ci(trans._('Unlink Layers'),),
    CommandId.LAYER_SELECT_LINKED: _ci(trans._('Select Linked Layers'),),
    CommandId.LAYER_CONVERT_TO_LABELS: _ci(trans._('Convert to Labels'),),
    CommandId.LAYER_CONVERT_TO_IMAGE: _ci(trans._('Convert to Image'),),
    CommandId.LAYER_CONVERT_TO_INT8: _ci(trans._('Convert to int8'),),
    CommandId.LAYER_CONVERT_TO_INT16: _ci(trans._('Convert to int16'),),
    CommandId.LAYER_CONVERT_TO_INT32: _ci(trans._('Convert to int32'),),
    CommandId.LAYER_CONVERT_TO_INT64: _ci(trans._('Convert to int64'),),
    CommandId.LAYER_CONVERT_TO_UINT8: _ci(trans._('Convert to uint8'),),
    CommandId.LAYER_CONVERT_TO_UINT16: _ci(trans._('Convert to uint16'),),
    CommandId.LAYER_CONVERT_TO_UINT32: _ci(trans._('Convert to uint32'),),
    CommandId.LAYER_CONVERT_TO_UINT64: _ci(trans._('Convert to uint64'),),
    CommandId.LAYER_PROJECT_MAX: _ci(trans._('Max projection'),),
    CommandId.LAYER_PROJECT_MIN: _ci(trans._('Min projection'),),
    CommandId.LAYER_PROJECT_STD: _ci(trans._('Std projection'),),
    CommandId.LAYER_PROJECT_SUM: _ci(trans._('Sum projection'),),
    CommandId.LAYER_PROJECT_MEAN: _ci(trans._('Mean projection'),),
    CommandId.LAYER_PROJECT_MEDIAN: _ci(trans._('Median projection'),),
}
# fmt: on
