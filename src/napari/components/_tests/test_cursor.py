from napari.components.cursor import Cursor


def test_cursor():
    """Test creating cursor object"""
    cursor = Cursor()
    assert cursor is not None
