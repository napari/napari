import sys
from typing import List

import pytest


@pytest.fixture
def _strict_qtbot(qtbot):
    """A modified qtbot fixture that makes sure no widgets have been leaked."""
    from qtpy.QtWidgets import QApplication

    initial = QApplication.topLevelWidgets()
    prior_exception = getattr(sys, 'last_value', None)

    yield qtbot

    # if an exception was raised during the test, we should just quit now and
    # skip looking for leaked widgets.
    if getattr(sys, 'last_value', None) is not prior_exception:
        return

    QApplication.processEvents()
    leaks = set(QApplication.topLevelWidgets()).difference(initial)
    # still not sure how to clean up some of the remaining vispy and qtconsole
    # vispy.app.backends._qt.CanvasBackendDesktop widgets...
    ignored_leaks = {'CanvasBackendDesktop', 'CompletionHtml', 'CallTipWidget'}
    leaks = {lk for lk in leaks if lk.__class__.__name__ not in ignored_leaks}
    if leaks:
        raise AssertionError(f'Widgets leaked!: {leaks}')


@pytest.fixture(scope="function")
def make_napari_viewer(_strict_qtbot, request):
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

    viewers: List[Viewer] = []

    def actual_factory(*model_args, viewer_class=Viewer, **model_kwargs):
        model_kwargs['show'] = model_kwargs.pop(
            'show', request.config.getoption("--show-viewer")
        )
        viewer = viewer_class(*model_args, **model_kwargs)
        viewers.append(viewer)
        return viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()
