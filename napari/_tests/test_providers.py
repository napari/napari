"""Test app-model providers.

Because `_provide_viewer` needs `_QtMainWindow` (otherwise returns `None`)
tests are here in `napari/_tests`, which are not run in headless mode.
"""
import pytest

from napari._app_model.injection._providers import _provide_viewer
from napari.utils._proxies import PublicOnlyProxy


def test_publicproxy_viewer(make_napari_viewer):
    """Test `_provide_viewer` outputs a `PublicOnlyProxy` when `Viewer` exists.

    Also check error raised when `Viewer` does not exist.
    """
    with pytest.raises(RuntimeError, match="No current `Viewer` found"):
        viewer = _provide_viewer()

    # Create a viewer
    make_napari_viewer()
    viewer = _provide_viewer()
    assert isinstance(viewer, PublicOnlyProxy)
