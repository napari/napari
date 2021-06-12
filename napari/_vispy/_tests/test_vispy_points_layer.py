import numpy as np
import pytest


@pytest.mark.parametrize("opacity", [(0), (0.3), (0.7), (1)])
def test_VispyPointsLayer(make_napari_viewer, opacity):
    """Test on the VispyPointsLayer object."""
    viewer = make_napari_viewer()
    points = np.array([[100, 100], [200, 200], [300, 100]])
    layer = viewer.add_points(points, size=30, opacity=opacity)
    visual = viewer.window.qt_viewer.layer_to_visual[layer]
    assert visual.node.opacity == opacity
