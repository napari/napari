"""
This module tests our "pytest plugin" made available in
``napari.utils._testsupport``.  It's here in the top level `_tests` folder
because it requires qt, and should be omitted from headless tests.
"""

import pytest

pytest_plugins = "pytester"


@pytest.mark.filterwarnings("ignore:`type` argument to addoption()::")
@pytest.mark.filterwarnings("ignore:The TerminalReporter.writer::")
def test_make_napari_viewer(pytester_pretty):
    """Make sure that our make_napari_viewer plugin works."""

    # create a temporary pytest test file
    pytester_pretty.makepyfile(
        """
        def test_make_viewer(make_napari_viewer):
            viewer = make_napari_viewer()
            assert viewer.layers == []
            assert viewer.__class__.__name__ == 'Viewer'
            assert not viewer.window._qt_window.isVisible()

    """
    )
    # run all tests with pytest
    result = pytester_pretty.runpytest()

    # check that all 1 test passed
    result.assert_outcomes(passed=1)
