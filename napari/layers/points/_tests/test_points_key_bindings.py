from napari.layers import Points
from napari.utils.action_manager import action_manager
from napari.utils.settings import SETTINGS


def test_select_all():
    """Test select all key binding."""
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    layer.mode = 'select'
    layer.class_keymap['Space'](layer)
    assert len(layer.selected_data) == 4

    # look for 'A' in Points default shortcuts
    actions = action_manager._get_layer_actions(Points).keys()
    shortcuts = [SETTINGS.shortcuts.shortcuts[name] for name in actions]
    assert ['A'] in shortcuts
