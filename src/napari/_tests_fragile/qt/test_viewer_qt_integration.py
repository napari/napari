import os

import pytest


@pytest.mark.skipif(os.environ.get('MIN_REQ', '0') == '1', reason='min req')
def test_qt_viewer_console_focus(qtbot, make_napari_viewer):
    """Test console has focus when instantiating from viewer."""
    viewer = make_napari_viewer(show=True)
    view = viewer.window._qt_viewer
    assert not view.console.hasFocus(), 'console has focus before being shown'

    view.toggle_console_visibility(None)

    def console_has_focus():
        assert view.console.hasFocus(), (
            'console does not have focus when shown'
        )

    qtbot.waitUntil(console_has_focus)
