from typing import List
import warnings

import pytest
from napari import Viewer


def pytest_addoption(parser):
    """An option to keep viewers hidden during tests.

    This speeds tests up by about %18, and does not seem to negatively affect
    tests.

    Example
    -------
    $ pytest napari -v --hide-viewer
    """

    parser.addoption(
        "--hide-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )


@pytest.fixture
def qapp(qapp):
    """A modified qapp fixture that makes sure no widgets have been leaked."""
    initial = qapp.topLevelWidgets()
    yield qapp
    qapp.processEvents()
    leaks = set(qapp.topLevelWidgets()).difference(initial)
    # still not sure how to clean up some of the remaining vispy
    # vispy.app.backends._qt.CanvasBackendDesktop widgets...
    if any([n.__class__.__name__ != 'CanvasBackendDesktop' for n in leaks]):
        raise AssertionError(f'Widgets leaked!: {leaks}')
    if leaks:
        warnings.warn(f'Widgets leaked!: {leaks}')


@pytest.fixture(scope="function")
def viewer_factory(qapp, request):
    viewers: List[Viewer] = []

    def actual_factory(*model_args, **model_kwargs):
        model_kwargs['show'] = model_kwargs.pop(
            'show', not request.config.getoption("--hide-viewer")
        )
        viewer = Viewer(*model_args, **model_kwargs)
        viewers.append(viewer)
        view = viewer.window.qt_viewer
        return view, viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()
