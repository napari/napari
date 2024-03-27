"""Test app-model providers.

Because `_provide_viewer` needs `_QtMainWindow` (otherwise returns `None`)
tests are here in `napari/_tests`, which are not run in headless mode.
"""

import pytest
from app_model.types import Action

from napari._app_model._app import get_app
from napari._app_model.injection._providers import (
    _provide_viewer,
    _provide_viewer_or_raise,
)
from napari.utils._proxies import PublicOnlyProxy
from napari.viewer import Viewer


def test_publicproxy_provide_viewer(capsys, make_napari_viewer):
    """Test `_provide_viewer` outputs a `PublicOnlyProxy` when appropriate.

    Check manual (e.g., internal) `_provide_viewer` calls can disable
    `PublicOnlyProxy` via `public_proxy` parameter but `PublicOnlyProxy` is always
    used when it is used as a provider.
    """
    # No current viewer, `None` should be returned
    viewer = _provide_viewer()
    assert viewer is None

    # Create a viewer
    make_napari_viewer()
    # Ensure we can disable via `public_proxy`
    viewer = _provide_viewer(public_proxy=False)
    assert isinstance(viewer, Viewer)

    # Ensure we get a `PublicOnlyProxy` when used as a provider
    def my_viewer(viewer: Viewer) -> Viewer:
        # Allows us to check type when `Action` executed
        print(type(viewer))

    action = Action(
        id='some.command.id',
        title='some title',
        callback=my_viewer,
    )
    app = get_app()
    app.register_action(action)
    app.commands.execute_command('some.command.id')
    captured = capsys.readouterr()
    assert 'napari.utils._proxies.PublicOnlyProxy' in captured.out


def test_provide_viewer_or_raise(make_napari_viewer):
    """Check `_provide_viewer_or_raise` raises or returns correct `Viewer`."""
    # raises when no viewer
    with pytest.raises(RuntimeError, match='No current `Viewer` found. test'):
        _provide_viewer_or_raise(msg='test')

    # create viewer
    make_napari_viewer()
    viewer = _provide_viewer_or_raise()
    assert isinstance(viewer, Viewer)

    viewer = _provide_viewer_or_raise(public_proxy=True)
    assert isinstance(viewer, PublicOnlyProxy)
