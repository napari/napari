from napari.utils.context import LayerListContextKeys


def test_layerlist_context(layer_list, points_layer):
    assert 'num_selected_layers' in LayerListContextKeys.__members__

    ctx = {}
    LLCK = LayerListContextKeys(ctx)
    assert LLCK.num_selected_layers == 0
    assert ctx['num_selected_layers'] == 0

    layer_list.selection.events.changed.connect(LLCK.update)
    layer_list.append(points_layer)
    assert LLCK.num_selected_layers == 1
    assert ctx['num_selected_layers'] == 1


def test_all_selected_layers_same_type(layer_list, points_layer, image_layer):
    assert 'all_selected_layers_same_type' in LayerListContextKeys.__members__

    ctx = {}
    LLCK = LayerListContextKeys(ctx)
    assert LLCK.all_selected_layers_same_type is False
    assert ctx['all_selected_layers_same_type'] is False

    layer_list.selection.events.changed.connect(LLCK.update)
    layer_list.append(points_layer)
    layer_list.append(points_layer)
    layer_list.append(image_layer)

    layer_list.selection = layer_list[:2]  # two points layers selected
    assert LLCK.all_selected_layers_same_type is True
    assert ctx['all_selected_layers_same_type'] is True

    layer_list.selection = [
        layer_list[0],
        layer_list[2],
    ]  # one points + one image
    assert LLCK.all_selected_layers_same_type is False
    assert ctx['all_selected_layers_same_type'] is False
