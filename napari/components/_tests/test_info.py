from napari.components.info import ViewerInfo


def test_info():
    """Test creating ViewerInfo object"""
    info = ViewerInfo()
    assert info is not None
    assert info.title == 'napari'
