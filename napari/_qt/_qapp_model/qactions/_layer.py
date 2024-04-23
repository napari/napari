import json

from app_model.expressions import parse_expression
from app_model.types import Action
from qtpy.QtWidgets import QApplication

from napari._app_model.constants import MenuGroup, MenuId
from napari.components import LayerList
from napari.layers import Layer
from napari.utils._numpy_json import NumpyEncoder
from napari.utils.notifications import show_warning
from napari.utils.translations import trans

__all__ = ('Q_LAYER_ACTIONS', 'is_valid_json_in_clipboard')


def _copy_spatial_to_clipboard(layer: Layer) -> None:
    json_data = {
        'scale': layer.scale,
        'translate': layer.translate,
    }

    d = json.dumps(json_data, cls=NumpyEncoder)

    clip = QApplication.clipboard()
    if clip is None:
        show_warning('Cannot access clipboard')
        return

    clip.setText(d)


def _copy_scale_to_clipboard(layer: Layer) -> None:
    json_data = {
        'scale': layer.scale,
    }

    d = json.dumps(json_data, cls=NumpyEncoder)

    clip = QApplication.clipboard()
    if clip is None:
        show_warning('Cannot access clipboard')
        return

    clip.setText(d)


def _paste_spatial_from_clipboard(ll: LayerList) -> None:
    clip = QApplication.clipboard()
    if clip is None:
        show_warning('Cannot access clipboard')
        return

    loaded = json.loads(clip.text())
    for layer in ll.selection:
        for key in loaded:
            setattr(layer, key, loaded[key])


def is_valid_json_in_clipboard():
    clip = QApplication.clipboard()
    if clip is None:
        return False

    try:
        data = json.loads(clip.text())
    except json.JSONDecodeError:
        return False

    return set(data).issubset({'scale', 'translate'})


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
        enablement=parse_expression('valid_spatial_json_clipboard'),
    ),
]
