"""Test label"""
from napari.components.overlays import TextOverlay


def test_label():
    """Test creating label object"""
    label = TextOverlay()
    assert label is not None
