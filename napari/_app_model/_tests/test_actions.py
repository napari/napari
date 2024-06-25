import pytest

from napari._app_model import get_app
from napari.components.layerlist import LayerList
from napari.layers import Points, Shapes


@pytest.mark.parametrize('layer_type', [Points, Shapes])
def test_check_duplicate_layer_action(layer_type):
    app = get_app()
    layer_list = LayerList()

    def _dummy():
        pass

    def provide_layer_list() -> 'LayerList':
        return layer_list

    with app.injection_store.register(providers=[(provide_layer_list,)]):
        layer_list.append(layer_type([], name='test'))
        layer_list.selection.active = layer_list[0]
        layer_list[0].events.data.connect(_dummy)
        assert len(layer_list[0].events.data.callbacks) == 2
        assert len(layer_list) == 1

        app.commands.execute_command('napari.layer.duplicate')

        assert len(layer_list) == 2
        assert layer_list[0].name == 'test'
        assert layer_list[1].name == 'test copy'
        assert layer_list[1].events.source is layer_list[1]
        assert (
            len(layer_list[1].events.data.callbacks) == 1
        )  # `events` Event Emitter
        assert layer_list[1].source.parent() is layer_list[0]
