import pytest

from napari.components.layerlist import LayerList
from napari.layers import Points
from napari.layers.layergroup import LayerGroup
from napari.utils.context import LayerListContextKeys


@pytest.mark.parametrize('LayersClass', [LayerList, LayerGroup])
def test_layerlist_context(LayersClass):
    assert 'layers_selection_count' in LayerListContextKeys.__members__

    ctx = {}
    llc = LayerListContextKeys(ctx)
    assert llc.layers_selection_count == 0
    assert ctx['layers_selection_count'] == 0

    layers = LayersClass()

    layers.selection.events.changed.connect(llc.update)
    layers.append(Points())
    assert llc.layers_selection_count == 1
    assert ctx['layers_selection_count'] == 1
