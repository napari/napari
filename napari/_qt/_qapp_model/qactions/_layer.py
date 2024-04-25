from __future__ import annotations

import json
import pickle

import numpy as np
from app_model.expressions import parse_expression
from app_model.types import Action
from qtpy.QtCore import QMimeData
from qtpy.QtWidgets import QApplication

from napari._app_model.constants import MenuGroup, MenuId
from napari.components import LayerList
from napari.layers import Layer
from napari.utils.notifications import show_warning
from napari.utils.translations import trans

__all__ = ('Q_LAYER_ACTIONS', 'is_valid_spatial_in_clipboard')


def _numpy_to_list(d: dict) -> dict:
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d


def _set_data_in_clipboard(data: dict) -> None:
    data = _numpy_to_list(data)
    clip = QApplication.clipboard()
    if clip is None:
        show_warning('Cannot access clipboard')
        return

    d = json.dumps(data)
    p = pickle.dumps(data)
    mime_data = QMimeData()
    mime_data.setText(d)
    mime_data.setData('application/octet-stream', p)

    clip.setMimeData(mime_data)


def _copy_spatial_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard(
        {
            'affine': layer.affine.linear_matrix,
            'rotate': layer.rotate,
            'scale': layer.scale,
            'shear': layer.shear,
            'translate': layer.translate,
        }
    )


def _copy_affine_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'affine': layer.affine.linear_matrix})


def _copy_rotate_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'rotate': layer.affine.linear_matrix})


def _copy_shear_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'shear': layer.scale})


def _copy_scale_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'scale': layer.scale})


def _copy_translate_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'translate': layer.scale})


def _get_spatial_from_clipboard() -> dict | None:
    clip = QApplication.clipboard()
    if clip is None:
        return None

    mime_data = clip.mimeData()
    if mime_data.data('application/octet-stream'):
        return pickle.loads(mime_data.data('application/octet-stream'))

    return json.loads(mime_data.text())


def _paste_spatial_from_clipboard(ll: LayerList) -> None:
    try:
        loaded = _get_spatial_from_clipboard()
    except (json.JSONDecodeError, pickle.UnpicklingError):
        show_warning('Cannot parse clipboard data')
        return
    if loaded is None:
        show_warning('Cannot access clipboard')
        return

    for layer in ll.selection:
        for key in loaded:
            setattr(layer, key, loaded[key])


def is_valid_spatial_in_clipboard() -> bool:
    try:
        loaded = _get_spatial_from_clipboard()
    except (json.JSONDecodeError, pickle.UnpicklingError):
        return False
    if not isinstance(loaded, dict):
        return False

    return set(loaded).issubset(
        {'affine', 'rotate', 'scale', 'shear', 'translate'}
    )


Q_LAYER_ACTIONS = [
    Action(
        id='napari.layer.copy_spatial_to_clipboard',
        title=trans._('Copy spatial to clipboard'),
        callback=_copy_spatial_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.copy_affine_to_clipboard',
        title=trans._('Copy affine to clipboard'),
        callback=_copy_affine_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.copy_rotate_to_clipboard',
        title=trans._('Copy rotate to clipboard'),
        callback=_copy_rotate_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.copy_scale_to_clipboard',
        title=trans._('Copy scale to clipboard'),
        callback=_copy_scale_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.copy_shear_to_clipboard',
        title=trans._('Copy shear to clipboard'),
        callback=_copy_shear_to_clipboard,
        menus=[{'id': MenuId.LAYERS_COPY_SPATIAL}],
    ),
    Action(
        id='napari.layer.copy_translate_to_clipboard',
        title=trans._('Copy translate to clipboard'),
        callback=_copy_translate_to_clipboard,
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
