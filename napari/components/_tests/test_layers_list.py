from napari.components import LayerList
from napari.layers import Image
import numpy as np


def test_empty_layers_list():
    """
    Test instantiating an empty LayerList object
    """
    layers = LayerList()

    assert len(layers) == 0


def test_initialize_from_list():
    layers = LayerList(
        [Image(np.random.random((10, 10))), Image(np.random.random((10, 10)))]
    )
    assert len(layers) == 2


def test_adding_layer():
    """
    Test adding a Layer
    """
    layers = LayerList()
    layer = Image(np.random.random((10, 10)))
    layers.append(layer)

    assert len(layers) == 1


def test_removing_layer():
    """
    Test removing a Layer
    """
    layers = LayerList()
    layer = Image(np.random.random((10, 10)))
    layers.append(layer)
    layers.remove(layer)

    assert len(layers) == 0


def test_indexing():
    """
    Test indexing into a LayerList
    """
    layers = LayerList()
    layer = Image(np.random.random((10, 10)), name='image')
    layers.append(layer)

    assert layers[0] == layer
    assert layers['image'] == layer


def test_insert():
    """
    Test inserting into a LayerList
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layers.append(layer_a)
    layers.insert(0, layer_b)

    assert list(layers) == [layer_b, layer_a]


def test_get_index():
    """
    Test getting indexing from LayerList
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layers.append(layer_a)
    layers.append(layer_b)

    assert layers.index(layer_a) == 0
    assert layers.index('image_a') == 0
    assert layers.index(layer_b) == 1
    assert layers.index('image_b') == 1


def test_reordering():
    """
    Test indexing into a LayerList by name
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layer_c = Image(np.random.random((15, 15)), name='image_c')
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)

    # Rearrange layers by tuple
    layers[:] = layers[(1, 0, 2)]
    assert list(layers) == [layer_b, layer_a, layer_c]

    # Swap layers by name
    layers['image_b', 'image_c'] = layers['image_c', 'image_b']
    assert list(layers) == [layer_c, layer_a, layer_b]

    # Reverse layers
    layers.reverse()
    assert list(layers) == [layer_b, layer_a, layer_c]


def test_naming():
    """
    Test unique naming in LayerList
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)), name='img')
    layer_b = Image(np.random.random((15, 15)), name='img')
    layers.append(layer_a)
    layers.append(layer_b)

    assert [l.name for l in layers] == ['img', 'img [1]']

    layer_b.name = 'chg'
    assert [l.name for l in layers] == ['img', 'chg']

    layer_a.name = 'chg'
    assert [l.name for l in layers] == ['chg [1]', 'chg']


def test_selection():
    """
    Test only last added is selected.
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)))
    layers.append(layer_a)
    assert layers[0].selected is True

    layer_b = Image(np.random.random((15, 15)))
    layers.append(layer_b)
    assert [l.selected for l in layers] == [False, True]

    layer_c = Image(np.random.random((15, 15)))
    layers.append(layer_c)
    assert [l.selected for l in layers] == [False] * 2 + [True]

    for l in layers:
        l.selected = True
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_d)
    assert [l.selected for l in layers] == [False] * 3 + [True]


def test_unselect_all():
    """
    Test unselecting
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)

    layers.unselect_all()
    assert [l.selected for l in layers] == [False] * 3

    for l in layers:
        l.selected = True
    layers.unselect_all(ignore=layer_b)
    assert [l.selected for l in layers] == [False, True, False]


def test_remove_selected():
    """
    Test removing selected layers
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)

    # remove last added layer as only one selected
    layers.remove_selected()
    assert list(layers) == [layer_a, layer_b]

    # check that the next to last layer is now selected
    assert [l.selected for l in layers] == [False, True]

    layers.remove_selected()
    assert list(layers) == [layer_a]
    assert [l.selected for l in layers] == [True]

    # select and remove first layer only
    layers.append(layer_b)
    layers.append(layer_c)
    assert list(layers) == [layer_a, layer_b, layer_c]
    layer_a.selected = True
    layer_b.selected = False
    layer_c.selected = False
    layers.remove_selected()
    assert list(layers) == [layer_b, layer_c]
    assert [l.selected for l in layers] == [True, False]

    # select and remove first and last layer of four
    layers.append(layer_a)
    layers.append(layer_d)
    assert list(layers) == [layer_b, layer_c, layer_a, layer_d]
    layer_a.selected = False
    layer_b.selected = True
    layer_c.selected = False
    layer_d.selected = True
    layers.remove_selected()
    assert list(layers) == [layer_c, layer_a]
    assert [l.selected for l in layers] == [True, False]

    # select and remove middle two layers of four
    layers.append(layer_b)
    layers.append(layer_d)
    layer_a.selected = True
    layer_b.selected = True
    layer_c.selected = False
    layer_d.selected = False
    layers.remove_selected()
    assert list(layers) == [layer_c, layer_d]
    assert [l.selected for l in layers] == [True, False]

    # select and remove all layers
    for l in layers:
        l.selected = True
    layers.remove_selected()
    assert len(layers) == 0


def test_move_selected():
    """
    Test removing selected layers
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)

    # Check nothing moves if given same insert and origin
    layers.unselect_all()
    layers.move_selected(2, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [False, False, True, False]

    # Move middle element to front of list and back
    layers.unselect_all()
    layers.move_selected(2, 0)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    assert [l.selected for l in layers] == [True, False, False, False]
    layers.unselect_all()
    layers.move_selected(0, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [False, False, True, False]

    # Move middle element to end of list and back
    layers.unselect_all()
    layers.move_selected(2, 3)
    assert list(layers) == [layer_a, layer_b, layer_d, layer_c]
    assert [l.selected for l in layers] == [False, False, False, True]
    layers.unselect_all()
    layers.move_selected(3, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [False, False, True, False]

    # Select first two layers only
    for l, s in zip(layers, [True, True, False, False]):
        l.selected = s
    # Move unselected middle element to front of list even if others selected
    layers.move_selected(2, 0)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    # Move selected first element back to middle of list
    layers.move_selected(0, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]

    # Select first two layers only
    for l, s in zip(layers, [True, True, False, False]):
        l.selected = s
    # Check nothing moves if given same insert and origin and multiple selected
    layers.move_selected(0, 0)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [True, True, False, False]

    # Check nothing moves if given same insert and origin and multiple selected
    layers.move_selected(1, 1)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [True, True, False, False]

    # Move first two selected to middle of list
    layers.move_selected(0, 1)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    assert [l.selected for l in layers] == [False, True, True, False]

    # Move middle selected to front of list
    layers.move_selected(2, 0)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [True, True, False, False]

    # Move first two selected to middle of list
    layers.move_selected(1, 2)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    assert [l.selected for l in layers] == [False, True, True, False]

    # Move middle selected to front of list
    layers.move_selected(1, 0)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert [l.selected for l in layers] == [True, True, False, False]

    # Select first and third layers only
    for l, s in zip(layers, [True, False, True, False]):
        l.selected = s
    # Move selection together to middle
    layers.move_selected(2, 2)
    assert list(layers) == [layer_b, layer_a, layer_c, layer_d]
    assert [l.selected for l in layers] == [False, True, True, False]
    layers[:] = layers[(1, 0, 2, 3)]

    # Move selection together to middle
    layers.move_selected(0, 1)
    assert list(layers) == [layer_b, layer_a, layer_c, layer_d]
    assert [l.selected for l in layers] == [False, True, True, False]
    layers[:] = layers[(1, 0, 2, 3)]

    # Move selection together to end
    layers.move_selected(2, 3)
    assert list(layers) == [layer_b, layer_d, layer_a, layer_c]
    assert [l.selected for l in layers] == [False, False, True, True]
    layers[:] = layers[(2, 0, 3, 1)]

    # Move selection together to end
    layers.move_selected(0, 2)
    assert list(layers) == [layer_b, layer_d, layer_a, layer_c]
    assert [l.selected for l in layers] == [False, False, True, True]
    layers[:] = layers[(2, 0, 3, 1)]

    # Move selection together to end
    layers.move_selected(0, 3)
    assert list(layers) == [layer_b, layer_d, layer_a, layer_c]
    assert [l.selected for l in layers] == [False, False, True, True]
    layers[:] = layers[(2, 0, 3, 1)]

    layer_e = Image(np.random.random((15, 15)))
    layer_f = Image(np.random.random((15, 15)))
    layers.append(layer_e)
    layers.append(layer_f)
    # Check current order is correct
    assert list(layers) == [
        layer_a,
        layer_b,
        layer_c,
        layer_d,
        layer_e,
        layer_f,
    ]
    # Select second and firth layers only
    for l, s in zip(layers, [False, True, False, False, True, False]):
        l.selected = s

    # Move selection together to middle
    layers.move_selected(1, 2)
    assert list(layers) == [
        layer_a,
        layer_c,
        layer_b,
        layer_e,
        layer_d,
        layer_f,
    ]
    assert [l.selected for l in layers] == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
