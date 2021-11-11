def test_layer_extent_cache(layer):
    layer.opacity = 1
    old_extent = layer.extent
    assert old_extent is layer.extent
    layer.opacity = 0.5
    assert old_extent is layer.extent

    layer.scale = (2,) + layer.scale[1:]
    old_extent2 = layer.extent
    assert old_extent is not layer.extent
    assert old_extent2 is layer.extent
    layer.opacity = 1
    assert old_extent2 is layer.extent
