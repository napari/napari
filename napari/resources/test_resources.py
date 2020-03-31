from napari.resources import import_resources
import os


def test_resources():
    """Test that we can build icons and resources."""
    out = import_resources(version='test')
    os.remove(out)
