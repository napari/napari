"""Qt 'Layer' menu Actions."""

from __future__ import annotations

import json
import pickle

import numpy as np
from app_model.expressions import parse_expression
from app_model.types import Action
from qtpy.QtCore import QMimeData
from qtpy.QtWidgets import QApplication

from napari._app_model.constants import MenuGroup, MenuId
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.components import LayerList
from napari.layers import Layer
from napari.utils.notifications import show_warning
from napari.utils.translations import trans

__all__ = ('Q_LAYERLIST_CONTEXT_ACTIONS', 'is_valid_spatial_in_clipboard')


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
            'affine': layer.affine.affine_matrix,
            'rotate': layer.rotate,
            'scale': layer.scale,
            'shear': layer.shear,
            'translate': layer.translate,
        }
    )


def _copy_affine_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'affine': layer.affine.affine_matrix})


def _copy_rotate_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'rotate': layer.rotate})


def _copy_shear_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'shear': layer.shear})


def _copy_scale_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'scale': layer.scale})


def _copy_translate_to_clipboard(layer: Layer) -> None:
    _set_data_in_clipboard({'translate': layer.translate})


def _get_spatial_from_clipboard() -> dict | None:
    clip = QApplication.clipboard()
    if clip is None:
        return None

    mime_data = clip.mimeData()
    if mime_data is None:  # pragma: no cover
        # we should never get here, but just in case
        return None
    if mime_data.data('application/octet-stream'):
        return pickle.loads(mime_data.data('application/octet-stream'))  # type: ignore[arg-type]

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
            loaded_attr_value = loaded[key]
            if isinstance(loaded_attr_value, list):
                loaded_attr_value = np.array(loaded_attr_value)
            if key == 'shear':
                loaded_attr_value = loaded_attr_value[
                    -(layer.ndim * (layer.ndim - 1)) // 2 :
                ]
            elif key == 'affine':
                loaded_attr_value = loaded_attr_value[
                    -(layer.ndim + 1) :, -(layer.ndim + 1) :
                ]
            elif isinstance(loaded_attr_value, np.ndarray):
                if loaded_attr_value.ndim == 1:
                    loaded_attr_value = loaded_attr_value[-layer.ndim :]
                elif loaded_attr_value.ndim == 2:
                    loaded_attr_value = loaded_attr_value[
                        -layer.ndim :, -layer.ndim :
                    ]

            setattr(layer, key, loaded_attr_value)


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


Q_LAYERLIST_CONTEXT_ACTIONS = [
    Action(
        id='napari.layer.copy_all_to_clipboard',
        title=trans._('Copy all to clipboard'),
        callback=_copy_spatial_to_clipboard,
        menus=[{'id': MenuId.LAYERS_CONTEXT_COPY_SPATIAL}],
        enablement=(LLSCK.num_selected_layers == 1),
    ),
    Action(
        id='napari.layer.copy_affine_to_clipboard',
        title=trans._('Copy affine to clipboard'),
        callback=_copy_affine_to_clipboard,
        menus=[{'id': MenuId.LAYERS_CONTEXT_COPY_SPATIAL}],
        enablement=(LLSCK.num_selected_layers == 1),
    ),
    Action(
        id='napari.layer.copy_rotate_to_clipboard',
        title=trans._('Copy rotate to clipboard'),
        callback=_copy_rotate_to_clipboard,
        menus=[{'id': MenuId.LAYERS_CONTEXT_COPY_SPATIAL}],
        enablement=(LLSCK.num_selected_layers == 1),
    ),
    Action(
        id='napari.layer.copy_scale_to_clipboard',
        title=trans._('Copy scale to clipboard'),
        callback=_copy_scale_to_clipboard,
        menus=[{'id': MenuId.LAYERS_CONTEXT_COPY_SPATIAL}],
        enablement=(LLSCK.num_selected_layers == 1),
    ),
    Action(
        id='napari.layer.copy_shear_to_clipboard',
        title=trans._('Copy shear to clipboard'),
        callback=_copy_shear_to_clipboard,
        menus=[{'id': MenuId.LAYERS_CONTEXT_COPY_SPATIAL}],
        enablement=(LLSCK.num_selected_layers == 1),
    ),
    Action(
        id='napari.layer.copy_translate_to_clipboard',
        title=trans._('Copy translate to clipboard'),
        callback=_copy_translate_to_clipboard,
        menus=[{'id': MenuId.LAYERS_CONTEXT_COPY_SPATIAL}],
        enablement=(LLSCK.num_selected_layers == 1),
    ),
    Action(
        id='napari.layer.paste_spatial_from_clipboard',
        title=trans._('Apply scale/transforms from Clipboard'),
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
