"""Test label"""
from napari.components.label import Label


def test_label():
    """Test creating label object"""
    label = Label()
    assert label is not None
