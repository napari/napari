import sys
import warnings
from typing import TYPE_CHECKING, List

import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest


def pytest_addoption(parser):
    parser.addoption(
        "--show-napari-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )


@pytest.fixture
def make_napari_viewer(qtbot, request: 'FixtureRequest'):
    """A fixture function that creates a napari viewer for use in testing.

    This uses a strict qtbot variant that asserts that no widgets are left over
    after the viewer is closed.

    Examples
    --------
    >>> def test_adding_shapes(make_napari_viewer):
    ...     viewer = make_napari_viewer()
    ...     viewer.add_shapes()
    ...     assert len(viewer.layers) == 1
    """
    from napari import Viewer
    from napari.utils.settings import SETTINGS

    SETTINGS.reset()
    viewers: List[Viewer] = []
    _strict = False

    from qtpy.QtWidgets import QApplication

    initial = QApplication.topLevelWidgets()
    prior_exception = getattr(sys, 'last_value', None)
    internal_test = request.module.__name__.startswith("napari.")

    def actual_factory(
        *model_args, ViewerClass=Viewer, strict=internal_test, **model_kwargs
    ):
        nonlocal _strict
        _strict = strict

        should_show = request.config.getoption("--show-napari-viewer")
        model_kwargs['show'] = model_kwargs.pop('show', should_show)

        viewer = ViewerClass(*model_args, **model_kwargs)
        viewers.append(viewer)

        return viewer

    yield actual_factory

    # Some tests might have the viewer closed, so this call will not be able
    # to access the window.
    try:
        SETTINGS.reset()
    except AttributeError:
        pass

    for viewer in viewers:
        viewer.close()

    # only check for leaked widgets if an exception was raised during the test,
    # or "strict" mode was not used.
    if _strict and getattr(sys, 'last_value', None) is prior_exception:
        QApplication.processEvents()
        leak = set(QApplication.topLevelWidgets()).difference(initial)
        # still not sure how to clean up some of the remaining vispy
        # vispy.app.backends._qt.CanvasBackendDesktop widgets...
        if any([n.__class__.__name__ != 'CanvasBackendDesktop' for n in leak]):
            # just a warning... but this can be converted to test errors
            # in pytest with `-W error`
            warnings.warn(f'Widgets leaked!: {leak}')
