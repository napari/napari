from typing import List
import warnings

import pytest
from napari import Viewer
from qtpy.QtWidgets import QApplication


def pytest_addoption(parser):
    """An option to show viewers during tests. (Hidden by default).

    Showing viewers decreases test speed by about %18.  Note, due to the
    placement of this conftest.py file, you must specify the napari folder (in
    the pytest command) to use this flag.

    Example
    -------
    $ pytest napari --show-viewer
    """
    parser.addoption(
        "--show-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )


@pytest.fixture
def qtbot(qtbot):
    """A modified qtbot fixture that makes sure no widgets have been leaked."""
    initial = QApplication.topLevelWidgets()
    yield qtbot
    QApplication.processEvents()
    leaks = set(QApplication.topLevelWidgets()).difference(initial)
    # still not sure how to clean up some of the remaining vispy
    # vispy.app.backends._qt.CanvasBackendDesktop widgets...
    if any([n.__class__.__name__ != 'CanvasBackendDesktop' for n in leaks]):
        raise AssertionError(f'Widgets leaked!: {leaks}')
    if leaks:
        warnings.warn(f'Widgets leaked!: {leaks}')


@pytest.fixture(scope="function")
def viewer_factory(qtbot, request):
    viewers: List[Viewer] = []

    def actual_factory(*model_args, **model_kwargs):
        model_kwargs['show'] = model_kwargs.pop(
            'show', request.config.getoption("--show-viewer")
        )
        viewer = Viewer(*model_args, **model_kwargs)
        viewers.append(viewer)
        view = viewer.window.qt_viewer
        return view, viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()
