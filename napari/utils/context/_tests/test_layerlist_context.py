from napari.components.layerlist import LayerList
from napari.layers import Points
from napari.utils.context import LayerListContextKeys


def test_layerlist_context():
    assert 'num_selected_layers' in LayerListContextKeys.__members__

    ctx = {}
    llc = LayerListContextKeys(ctx)
    assert llc.num_selected_layers == 0
    assert ctx['num_selected_layers'] == 0

    layers = LayerList()

    layers.selection.events.changed.connect(llc.update)
    layers.append(Points())
    assert llc.num_selected_layers == 1
    assert ctx['num_selected_layers'] == 1
