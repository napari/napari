from napari.components.grid import GridCanvas


# build mock layers with visible attribute
class mock_layer:
    def __init__(self, visible=True):
        self.visible = visible


layers_9 = [mock_layer() for _ in range(9)]
layers_5 = [mock_layer() for _ in range(5)]
layers_7 = [mock_layer() for _ in range(7)]
layers_3 = [mock_layer() for _ in range(3)]
layers_10 = [mock_layer() for _ in range(10)]

# build mock layers with invisible attribute
layers_9_invisible = [mock_layer(visible=False) for _ in range(9)]
layers_5_invisible = [mock_layer(visible=False) for _ in range(5)]
layers_7_invisible = [mock_layer(visible=False) for _ in range(7)]
layers_3_invisible = [mock_layer(visible=False) for _ in range(3)]
layers_10_invisible = [mock_layer(visible=False) for _ in range(10)]


def test_grid_creation():
    """Test creating grid object"""
    grid = GridCanvas()
    assert grid is not None
    assert not grid.enabled
    assert grid.shape == (-1, -1)
    assert grid.stride == 1
    assert grid.spacing == 0


def test_shape_stride_creation():
    """Test creating grid object"""
    grid = GridCanvas(shape=(3, 4), stride=2)
    assert grid.shape == (3, 4)
    assert grid.stride == 2


def test_actual_shape_and_position():
    """Test actual shape"""
    grid = GridCanvas(enabled=True)
    assert grid.enabled

    # 9 layers get put in a (3, 3) grid
    assert grid.actual_shape(layers_9) == (3, 3)
    assert grid.position(0, layers_9) == (0, 0)
    assert grid.position(2, layers_9) == (0, 2)
    assert grid.position(3, layers_9) == (1, 0)
    assert grid.position(8, layers_9) == (2, 2)

    # 5 layers get put in a (2, 3) grid
    assert grid.actual_shape(layers_5) == (2, 3)
    assert grid.position(0, layers_5) == (0, 0)
    assert grid.position(2, layers_5) == (0, 2)
    assert grid.position(3, layers_5) == (1, 0)

    # 10 layers get put in a (3, 4) grid
    assert grid.actual_shape(layers_10) == (3, 4)
    assert grid.position(0, layers_10) == (0, 0)
    assert grid.position(2, layers_10) == (0, 2)
    assert grid.position(3, layers_10) == (0, 3)
    assert grid.position(8, layers_10) == (2, 0)


def test_actual_shape_with_stride():
    """Test actual shape"""
    grid = GridCanvas(enabled=True, stride=2)
    assert grid.enabled

    # 7 layers get put in a (2, 2) grid
    assert grid.actual_shape(layers_7) == (2, 2)
    assert grid.position(0, layers_7) == (0, 0)
    assert grid.position(1, layers_7) == (0, 0)
    assert grid.position(2, layers_7) == (0, 1)
    assert grid.position(3, layers_7) == (0, 1)
    assert grid.position(6, layers_7) == (1, 1)

    # 3 layers get put in a (1, 2) grid
    assert grid.actual_shape(layers_3) == (1, 2)
    assert grid.position(0, layers_3) == (0, 0)
    assert grid.position(1, layers_3) == (0, 0)
    assert grid.position(2, layers_3) == (0, 1)


def test_actual_shape_and_position_negative_stride():
    """Test actual shape"""
    grid = GridCanvas(enabled=True, stride=-1)
    assert grid.enabled

    # 9 layers get put in a (3, 3) grid
    assert grid.actual_shape(layers_9) == (3, 3)
    assert grid.position(0, layers_9) == (2, 2)
    assert grid.position(2, layers_9) == (2, 0)
    assert grid.position(3, layers_9) == (1, 2)
    assert grid.position(8, layers_9) == (0, 0)


def test_actual_shape_grid_disabled():
    """Test actual shape with grid disabled"""
    grid = GridCanvas()
    assert not grid.enabled
    assert grid.actual_shape(layers_9) == (1, 1)
    assert grid.position(3, layers_9) == (0, 0)
