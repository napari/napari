import numpy as np

from napari.utils import nbscreenshot


def test_nbscreenshot(make_napari_viewer):
    """Test taking a screenshot."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)

    rich_display_object = nbscreenshot(viewer)
    assert hasattr(rich_display_object, '_repr_png_')
    # Trigger method that would run in jupyter notebook cell automatically
    rich_display_object._repr_png_()
    assert rich_display_object.image is not None
