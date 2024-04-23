import json

from app_model.types import Action
from qtpy.QtWidgets import QApplication

from napari._app_model.constants import MenuGroup, MenuId
from napari.components import LayerList
from napari.layers import Layer
from napari.utils._numpy_json import NumpyEncoder
from napari.utils.translations import trans


def _copy_spatial_to_clipboard(layer: Layer) -> None:
    json_data = {
        'scale': layer.scale,
        'translate': layer.translate,
    }

    d = json.dumps(json_data, cls=NumpyEncoder)

    QApplication.clipboard().setText(d)


def _copy_scale_to_clipboard(layer: Layer) -> None:
    json_data = {
        'scale': layer.scale,
    }

    d = json.dumps(json_data, cls=NumpyEncoder)

    QApplication.clipboard().setText(d)


def _paste_spatial_from_clipboard(ll: LayerList) -> None:
    QApplication.clipboard().text()
    loaded = json.loads(QApplication.clipboard().text())
    for layer in ll.selection:
        for key in loaded:
            setattr(layer, key, loaded[key])


Q_LAYER_ACTIONS = [
    Action(
        id='napari.layer.copy_spatial_to_clipboard',
        title=trans._('Copy spatial to clipboard'),
        callback=_copy_spatial_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.copy_scale_to_clipboard',
        title=trans._('Copy scale to clipboard'),
        callback=_copy_scale_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.paste_spatial_from_clipboard',
        title=trans._('Paste Spatial from Clipboard'),
        callback=_paste_spatial_from_clipboard,
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.LAYERLIST_CONTEXT.COPY_SPATIAL,
            }
        ],
    ),
]
