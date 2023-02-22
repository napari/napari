import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMainWindow

from napari._qt.utils import (
    QBYTE_FLAG,
    add_flash_animation,
    is_qbyte,
    qbytearray_to_str,
    qt_might_be_rich_text,
    qt_signals_blocked,
    str_to_qbytearray,
)
from napari.utils._proxies import PublicOnlyProxy


class Emitter(QObject):
    test_signal = Signal()

    def go(self):
        self.test_signal.emit()


def test_signal_blocker(qtbot):
    """make sure context manager signal blocker works"""
    import pytestqt.exceptions

    obj = Emitter()

    # make sure signal works
    with qtbot.waitSignal(obj.test_signal):
        obj.go()

    # make sure blocker works
    with qt_signals_blocked(obj), pytest.raises(
        pytestqt.exceptions.TimeoutError
    ), qtbot.waitSignal(obj.test_signal, timeout=500):
        obj.go()


def test_is_qbyte_valid():
    is_qbyte(QBYTE_FLAG)
    is_qbyte(
        "!QBYTE_AAAA/wAAAAD9AAAAAgAAAAAAAAECAAACePwCAAAAAvsAAAAcAGwAYQB5AGUAcgAgAGMAbwBuAHQAcgBvAGwAcwEAAAAAAAABFwAAARcAAAEX+wAAABQAbABhAHkAZQByACAAbABpAHMAdAEAAAEXAAABYQAAALcA////AAAAAwAAAAAAAAAA/AEAAAAB+wAAAA4AYwBvAG4AcwBvAGwAZQAAAAAA/////wAAADIA////AAADPAAAAngAAAAEAAAABAAAAAgAAAAI/AAAAAA="
    )


def test_str_to_qbytearray_valid():
    with pytest.raises(ValueError):
        str_to_qbytearray("")

    with pytest.raises(ValueError):
        str_to_qbytearray("FOOBAR")

    with pytest.raises(ValueError):
        str_to_qbytearray(
            "_AAAA/wAAAAD9AAAAAgAAAAAAAAECAAACePwCAAAAAvsAAAAcAGwAYQB5AGUAcgAgAGMAbwBuAHQAcgBvAGwAcwEAAAAAAAABFwAAARcAAAEX+wAAABQAbABhAHkAZQByACAAbABpAHMAdAEAAAEXAAABYQAAALcA////AAAAAwAAAAAAAAAA/AEAAAAB+wAAAA4AYwBvAG4AcwBvAGwAZQAAAAAA/////wAAADIA////AAADPAAAAngAAAAEAAAABAAAAAgAAAAI/AAAAAA="
        )


def test_str_to_qbytearray_invalid():
    with pytest.raises(ValueError):
        str_to_qbytearray("")

    with pytest.raises(ValueError):
        str_to_qbytearray(
            "_AAAA/wAAAAD9AAAAAgAAAAAAAAECAAACePwCAAAAAvsAAAAcAGwAYQB5AGUAcgAgAGMAbwBuAHQAcgBvAGwAcwEAAAAAAAABFwAAARcAAAEX+wAAABQAbABhAHkAZQByACAAbABpAHMAdAEAAAEXAAABYQAAALcA////AAAAAwAAAAAAAAAA/AEAAAAB+wAAAA4AYwBvAG4AcwBvAGwAZQAAAAAA/////wAAADIA////AAADPAAAAngAAAAEAAAABAAAAAgAAAAI/AAAAAA="
        )


def test_qbytearray_to_str(qtbot):
    widget = QMainWindow()
    qtbot.addWidget(widget)

    qbyte = widget.saveState()
    qbyte_string = qbytearray_to_str(qbyte)
    assert is_qbyte(qbyte_string)


def test_qbytearray_to_str_and_back(qtbot):
    widget = QMainWindow()
    qtbot.addWidget(widget)

    qbyte = widget.saveState()
    assert str_to_qbytearray(qbytearray_to_str(qbyte)) == qbyte


def test_add_flash_animation(qtbot):
    widget = QMainWindow()
    qtbot.addWidget(widget)
    assert widget.graphicsEffect() is None
    add_flash_animation(widget)
    assert widget.graphicsEffect() is not None
    assert hasattr(widget, "_flash_animation")
    qtbot.wait(350)
    assert widget.graphicsEffect() is None
    assert not hasattr(widget, "_flash_animation")


def test_qt_might_be_rich_text(qtbot):
    widget = QMainWindow()
    qtbot.addWidget(widget)
    assert qt_might_be_rich_text("<b>rich text</b>")
    assert not qt_might_be_rich_text("plain text")


def test_thread_proxy_guard(monkeypatch, qapp, single_threaded_executor):
    class X:
        a = 1

    monkeypatch.setenv('NAPARI_ENSURE_PLUGIN_MAIN_THREAD', 'True')

    x = X()
    x_proxy = PublicOnlyProxy(x)

    f = single_threaded_executor.submit(x.__setattr__, 'a', 2)
    f.result()
    assert x.a == 2

    f = single_threaded_executor.submit(x_proxy.__setattr__, 'a', 3)
    with pytest.raises(RuntimeError):
        f.result()
    assert x.a == 2
