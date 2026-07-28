import pytest

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


def test_hidden_layers_excluded_with_stride_one():
    """Test hidden layers are excluded from grid layout with stride=1 (default)."""
    grid = GridCanvas(enabled=True)
    layers_mixed = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
    ]
    # Only 3 visible layers → ceil(sqrt(3)) = 2 columns, ceil(3/2) = 2 rows → (2, 2)
    assert grid.actual_shape(layers_mixed) == (2, 2)

    # Visible layers get grid positions
    assert grid.position(0, layers_mixed) == (0, 0)
    assert grid.position(2, layers_mixed) == (0, 1)
    assert grid.position(4, layers_mixed) == (1, 0)

    # Hidden layers return (-1, -1)
    assert grid.position(1, layers_mixed) == (-1, -1)
    assert grid.position(3, layers_mixed) == (-1, -1)


def test_hidden_layers_negative_stride():
    """Test hidden layers with negative stride."""
    grid = GridCanvas(enabled=True, stride=-1)
    layers_mixed = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
    ]
    # 2 visible layers → (1, 2)
    assert grid.actual_shape(layers_mixed) == (1, 2)
    # With negative stride, visible layer 0 (first) is placed last
    assert grid.position(0, layers_mixed) == (0, 1)
    assert grid.position(1, layers_mixed) == (-1, -1)
    assert grid.position(2, layers_mixed) == (0, 0)


def test_all_layers_hidden():
    """Test when all layers are hidden with stride=1."""
    grid = GridCanvas(enabled=True)
    layers_all_hidden = [mock_layer(visible=False) for _ in range(5)]
    # No visible layers → (1, 1)
    assert grid.actual_shape(layers_all_hidden) == (1, 1)
    for i in range(5):
        assert grid.position(i, layers_all_hidden) == (-1, -1)


def test_hidden_layers_with_large_stride():
    """Test with stride >= 2 where ALL layers are counted regardless of visibility."""
    grid = GridCanvas(enabled=True, stride=3)
    layers_mixed = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
        mock_layer(visible=False),
    ]
    # stride=3 → ceil(4/3)=2 grid squares → (1, 2) shape
    assert grid.actual_shape(layers_mixed) == (1, 1)
    # Every layer gets a grid position (visibility ignored for stride >= 2)
    # Stride=3 packs 3 layers in cell 0, then 1 layer in cell 1
    assert grid.position(0, layers_mixed) == (0, 0)
    assert grid.position(1, layers_mixed) == (-1, -1)
    assert grid.position(2, layers_mixed) == (0, 0)
    assert grid.position(3, layers_mixed) == (-1, -1)


def test_contents_at_with_hidden_layers():
    """Test contents_at excludes hidden layers."""
    grid = GridCanvas(enabled=True)
    layers_mixed = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
    ]
    # 2 visible layers → (1, 2)
    assert grid.contents_at((0, 0), layers_mixed) == (0,)
    assert grid.contents_at((0, 1), layers_mixed) == (2,)
    # Position that doesn't exist in the grid → ()
    assert grid.contents_at((1, 0), layers_mixed) == ()


def test_iter_viewboxes_with_hidden_layers():
    """Test iter_viewboxes excludes hidden layers from contents."""
    grid = GridCanvas(enabled=True)
    layers_mixed = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
    ]
    viewboxes = dict(grid.iter_viewboxes(layers_mixed))
    # (1, 2) grid for 2 visible layers
    assert len(viewboxes) == 2
    assert viewboxes[(0, 0)] == (0,)
    assert viewboxes[(0, 1)] == (2,)


def test_grid_position_out_of_bounds():
    """Test position with out-of-bounds index raises ValueError."""
    grid = GridCanvas(enabled=True)
    layers = [mock_layer() for _ in range(3)]
    with pytest.raises(ValueError, match='Index 5 is out of bounds'):
        grid.position(5, layers)
    with pytest.raises(ValueError, match='Index -1 is out of bounds'):
        grid.position(-1, layers)


def test_grid_with_empty_layers():
    """Test grid methods with empty layers list."""
    grid = GridCanvas(enabled=True)
    assert grid.actual_shape([]) == (1, 1)
    assert grid.position(0, []) == (0, 0)
    assert grid.contents_at((0, 0), []) == ()
    # Empty layers still result in a (1, 1) grid with one empty viewbox
    assert dict(grid.iter_viewboxes([])) == {(0, 0): ()}


def test_effective_indices():
    """Test _effective_indices returns correct active indices."""
    grid = GridCanvas(enabled=True)
    layers_mixed = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
    ]
    # stride=1 → only visible layers
    assert grid._effective_indices(layers_mixed) == [0, 2]
    # stride >= 2 → all layers regardless of visibility
    grid.stride = 2
    assert grid._effective_indices(layers_mixed) == [0, 2]
    grid.stride = 3
    assert grid._effective_indices(layers_mixed) == [0, 2]
    # Empty layers → empty list
    assert grid._effective_indices([]) == []
    # None → empty list
    assert grid._effective_indices() == []

def test_hidden_layers_with_stride_equal_visible_count():
    """Test stride=n with n visible + invisible layers ->  shape (1, 1).
    
    Regression test: stride=n with exactly n visible layers and
    one or more invisible layers should produce a single viewbox.
    """
    grid = GridCanvas(enabled=True, stride=2)
    layers = [
        mock_layer(visible=True),
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=False),
    ]
    # Both visible layers are in group 0 (indices 0,1)
    # Group 1 (index 2) is entirely invisible -> not counted
    assert grid.actual_shape(layers) == (1, 1)
    assert grid.position(0, layers) == (0, 0)
    assert grid.position(1, layers) == (0, 0)
    assert grid.position(2, layers) == (-1, -1)

    # Alternate visibility test
    layers_alt = [
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
        mock_layer(visible=False),
        mock_layer(visible=True),
        mock_layer(visible=False),
    ]
  
    assert grid.actual_shape(layers_alt) == (2, 2)
    assert grid.position(0, layers_alt) == (0, 0)
    assert grid.position(1, layers_alt) == (-1, -1)
    assert grid.position(2, layers_alt) == (0, 1)
    assert grid.position(3, layers_alt) == (-1, -1)
    assert grid.position(4, layers_alt) == (1, 0)
    assert grid.position(5, layers_alt) == (-1, -1)