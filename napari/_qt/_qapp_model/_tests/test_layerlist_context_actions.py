import pytest

from napari._app_model import get_app
from napari._app_model.actions._layerlist_context_actions import (
    LAYERLIST_CONTEXT_ACTIONS,
)


@pytest.mark.parametrize('layer_action', LAYERLIST_CONTEXT_ACTIONS)
def test_layer_actions_ctx_menu_execute_command(
    layer_action, make_napari_viewer
):
    app = get_app()
    make_napari_viewer()
    command_id = layer_action.id

    if command_id == 'napari.layer.merge_stack':
        with pytest.raises(IndexError, match=r'images list is empty'):
            app.commands.execute_command(command_id)
    elif command_id in [
        'napari.layer.link_selected_layers',
        'napari.layer.unlink_selected_layers',
    ]:
        with pytest.raises(ValueError, match=r'at least one'):
            app.commands.execute_command(command_id)
    else:
        app.commands.execute_command(command_id)
