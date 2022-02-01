import numpy as np

from napari._qt.widgets.qt_action_context_menu import QtActionContextMenu
from napari.components.layerlist import LayerList
from napari.layers import Image, Labels
from napari.layers._layer_actions import _LAYER_ACTIONS
from napari.utils.context._expressions import parse_expression


def test_action_menu(qapp):
    ACTIONS = {
        'add_one': {
            'description': 'Add one',
            'action': lambda x: x.append(1),
            'enable_when': parse_expression('count == 0'),
        }
    }

    menu = QtActionContextMenu(ACTIONS)
    menu.update_from_context({'count': 0, 'is_ready': True})
    assert menu._get_action('add_one').isEnabled()
    assert menu._get_action('add_one').isVisible()

    CONTEXT_KEYS = {
        'count': lambda x: len(x),
        'is_ready': lambda x: True,
    }
    some_object = [42]
    ctx = {k: v(some_object) for k, v in CONTEXT_KEYS.items()}
    menu.update_from_context(ctx)
    assert not menu._get_action('add_one').isEnabled()


def test_layer_action_menu(qapp):
    """Test the actions in LAYER_ACTIONS."""
    menu = QtActionContextMenu(_LAYER_ACTIONS)
    layer_list = LayerList([])
    menu.update_from_context(layer_list._ctx)
    assert not menu._get_action('napari:convert_to_image').isEnabled()

    layer_list.append(Labels(np.zeros((8, 8), int)))
    menu.update_from_context(layer_list._ctx)
    assert menu._get_action('napari:convert_to_image').isEnabled()
    assert not menu._get_action('napari:convert_to_labels').isEnabled()

    layer_list.append(Image(np.zeros((8, 8))))
    menu.update_from_context(layer_list._ctx)
    assert not menu._get_action('napari:convert_to_image').isEnabled()
    assert menu._get_action('napari:convert_to_labels').isEnabled()
    assert not menu._get_action('napari:link_selected_layers').isEnabled()

    layer_list.select_all()
    menu.update_from_context(layer_list._ctx)
    assert menu._get_action('napari:link_selected_layers').isEnabled()
    assert not menu._get_action('napari:unlink_selected_layers').isEnabled()

    layer_list.link_layers()
    menu.update_from_context(layer_list._ctx)
    assert not menu._get_action('napari:link_selected_layers').isEnabled()
    assert menu._get_action('napari:unlink_selected_layers').isEnabled()
