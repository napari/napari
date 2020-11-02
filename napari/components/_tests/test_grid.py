from napari.components.grid import Grid


def test_grid():
    """Test creating grid object"""
    grid = Grid()
    assert grid is not None
