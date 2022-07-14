from enum import Enum


class MenuId(str, Enum):
    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value


class MenuGroup:
    class LAYERLIST_CONTEXT:
        NAVIGATION = 'navigation'
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'
