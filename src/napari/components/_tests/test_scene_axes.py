from napari.components.overlays.scene_axes import SceneAxesOverlay


def test_axes():
    """Test creating axes object"""
    axes = SceneAxesOverlay()
    assert axes is not None
