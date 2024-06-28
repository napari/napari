import pytest

from napari._app_model import get_app
from napari._app_model.actions._layer_actions import LAYER_ACTIONS
from napari.components.layerlist import LayerList


@pytest.mark.parametrize('layer_action', LAYER_ACTIONS)
def test_layer_actions_ctx_menu_execute_command(layer_action):
    app = get_app()
    command_id = layer_action.id

    def provide_layer_list() -> 'LayerList':
        return LayerList()

    with app.injection_store.register(providers=[(provide_layer_list,)]):
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
