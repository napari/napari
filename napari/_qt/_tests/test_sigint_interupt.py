import os

import pytest
from qtpy.QtCore import QTimer

from napari._qt.utils import _maybe_allow_interrupt


@pytest.fixture
def platform_simulate_ctrl_c():
    import signal
    from functools import partial

    if hasattr(signal, "CTRL_C_EVENT"):
        win32api = pytest.importorskip('win32api')
        return partial(win32api.GenerateConsoleCtrlEvent, 0, 0)
    else:
        # we're not on windows
        return partial(os.kill, os.getpid(), signal.SIGINT)


@pytest.mark.skipif(os.name != "Windows", reason="Windows specific")
def test_sigint(qapp, platform_simulate_ctrl_c, make_napari_viewer):
    def fire_signal():
        platform_simulate_ctrl_c()

    make_napari_viewer()
    QTimer.singleShot(100, fire_signal)
    with pytest.raises(KeyboardInterrupt):
        with _maybe_allow_interrupt(qapp):
            qapp.exec_()
