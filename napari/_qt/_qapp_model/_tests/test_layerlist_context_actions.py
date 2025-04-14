import pytest

from napari._app_model import get_app_model
from napari._app_model.actions._layerlist_context_actions import (
    LAYERLIST_CONTEXT_ACTIONS,
)


@pytest.mark.parametrize('layer_action', LAYERLIST_CONTEXT_ACTIONS)
def test_layer_actions_ctx_menu_execute_command(
    layer_action, make_napari_viewer
):
    """
    Test layer context menu actions via app-model `execute_command`.

    Note:
        This test is here only to ensure app-model action dispatch mechanism
        is working for these actions (which use the `_provide_active_layer_list` provider).

        To check a set of functional tests related to these actions you can
        see: https://github.com/napari/napari/blob/main/napari/layers/_tests/test_layer_actions.py
    """
    app = get_app_model()
    make_napari_viewer()
    command_id = layer_action.id

    if command_id == 'napari.layer.merge_stack':
        with pytest.raises(IndexError, match=r'images list is empty'):
            app.commands.execute_command(command_id)
    elif command_id == 'napari.layer.merge_rgb':
        with pytest.raises(
            ValueError, match='Merging to RGB requires exactly 3 Image'
        ):
            app.commands.execute_command(command_id)
    elif command_id in [
        'napari.layer.link_selected_layers',
        'napari.layer.unlink_selected_layers',
    ]:
        with pytest.raises(ValueError, match=r'at least one'):
            app.commands.execute_command(command_id)
    else:
        app.commands.execute_command(command_id)
