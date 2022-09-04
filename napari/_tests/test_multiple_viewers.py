import gc
from unittest.mock import patch

import pytest

from napari import Viewer


# @pytest.mark.skip(reason="problem with clean")
def test_multi_viewers_dont_clash(qtbot):
    v1 = Viewer(show=False, title='v1')
    v2 = Viewer(show=False, title='v2')
    assert not v1.grid.enabled
    assert not v2.grid.enabled

    v1.window.activate()  # a click would do this in the actual gui
    v1.window._qt_viewer.viewerButtons.gridViewButton.click()

    assert not v2.grid.enabled
    assert v1.grid.enabled

    with patch.object(v1.window._qt_window, '_save_current_window_settings'):
        v1.close()
    with patch.object(v2.window._qt_window, '_save_current_window_settings'):
        v2.close()
    qtbot.wait(50)
    gc.collect()
