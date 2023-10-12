from napari._app_model.injection._providers import _provide_viewer
from napari.utils._proxies import PublicOnlyProxy


def test_publicproxy_viewer(make_napari_viewer):
    """Test `_provide_viewer` outputs a `PublicOnlyProxy` when viewer exists."""
    viewer = _provide_viewer()
    assert viewer is None

    # Create a viewer
    make_napari_viewer()
    viewer = _provide_viewer()
    assert isinstance(viewer, PublicOnlyProxy)
