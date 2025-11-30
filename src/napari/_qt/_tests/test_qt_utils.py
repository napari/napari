from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from qtpy.QtCore import QByteArray, QObject, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog, QMainWindow

from napari._qt.utils import (
    QBYTE_FLAG,
    add_flash_animation,
    get_color,
    is_qbyte,
    qbytearray_to_str,
    qt_might_be_rich_text,
    qt_signals_blocked,
    str_to_qbytearray,
)
from napari.utils._proxies import PublicOnlyProxy

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


class Emitter(QObject):
    test_signal = Signal()

    def go(self):
        self.test_signal.emit()


def test_signal_blocker(qtbot):
    """make sure context manager signal blocker works"""

    obj = Emitter()

    # make sure signal works
    with qtbot.waitSignal(obj.test_signal):
        obj.go()

    # make sure blocker works
    with (
        qt_signals_blocked(obj),
        qtbot.assert_not_emitted(obj.test_signal, wait=500),
    ):
        obj.go()
    obj.deleteLater()


def test_is_qbyte_valid():
    assert is_qbyte(QBYTE_FLAG)
    assert is_qbyte(
        '!QBYTE_AAAA/wAAAAD9AAAAAgAAAAAAAAECAAACePwCAAAAAvsAAAAcAGwAYQB5AGUAcgAgAGMAbwBuAHQAcgBvAGwAcwEAAAAAAAABFwAAARcAAAEX+wAAABQAbABhAHkAZQByACAAbABpAHMAdAEAAAEXAAABYQAAALcA////AAAAAwAAAAAAAAAA/AEAAAAB+wAAAA4AYwBvAG4AcwBvAGwAZQAAAAAA/////wAAADIA////AAADPAAAAngAAAAEAAAABAAAAAgAAAAI/AAAAAA='
    )


def test_str_to_qbytearray_valid():
    assert isinstance(
        str_to_qbytearray(
            '!QBYTE_AAAA/wAAAAD9AAAAAgAAAAAAAAECAAACePwCAAAAAvsAAAAcAGwAYQB5AGUAcgAgAGMAbwBuAHQAcgBvAGwAcwEAAAAAAAABFwAAARcAAAEX+wAAABQAbABhAHkAZQByACAAbABpAHMAdAEAAAEXAAABYQAAALcA////AAAAAwAAAAAAAAAA/AEAAAAB+wAAAA4AYwBvAG4AcwBvAGwAZQAAAAAA/////wAAADIA////AAADPAAAAngAAAAEAAAABAAAAAgAAAAI/AAAAAA='
        ),
        QByteArray,
    )


def test_str_to_qbytearray_invalid():
    with pytest.raises(ValueError, match='Invalid QByte string'):
        str_to_qbytearray('')

    with pytest.raises(ValueError, match='Invalid QByte string'):
        str_to_qbytearray('FOOBAR')

    with pytest.raises(ValueError, match='Invalid QByte string'):
        str_to_qbytearray(
            '_AAAA/wAAAAD9AAAAAgAAAAAAAAECAAACePwCAAAAAvsAAAAcAGwAYQB5AGUAcgAgAGMAbwBuAHQAcgBvAGwAcwEAAAAAAAABFwAAARcAAAEX+wAAABQAbABhAHkAZQByACAAbABpAHMAdAEAAAEXAAABYQAAALcA////AAAAAwAAAAAAAAAA/AEAAAAB+wAAAA4AYwBvAG4AcwBvAGwAZQAAAAAA/////wAAADIA////AAADPAAAAngAAAAEAAAABAAAAAgAAAAI/AAAAAA='
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
    assert hasattr(widget, '_flash_animation')
    qtbot.wait(350)
    assert widget.graphicsEffect() is None
    assert not hasattr(widget, '_flash_animation')


@pytest.mark.usefixtures('qapp')
def test_qt_might_be_rich_text():
    assert qt_might_be_rich_text('<b>rich text</b>')
    assert not qt_might_be_rich_text('plain text')


@pytest.mark.usefixtures('qapp')
def test_thread_proxy_guard(monkeypatch, single_threaded_executor):
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


def _assert_eq(p1: Any, p2: Any, text: str = '') -> None:
    if isinstance(p2, np.ndarray):
        np.testing.assert_array_equal(p1, p2, err_msg=text)
    else:
        assert p1 == p2, text


DEFAULT_COLOR_HEX = '#ffffff'
DEFAULT_COLOR_ARRAY = np.asarray([1, 1, 1])
DEFAULT_COLOR_QCOLOR = QColor(255, 255, 255)


@pytest.mark.parametrize(
    ('color', 'mode', 'expected'),
    [
        (None, 'hex', DEFAULT_COLOR_HEX),
        ('#FF00FF', 'hex', '#ff00ff'),
        (None, 'array', DEFAULT_COLOR_ARRAY),
        (np.asarray([255, 0, 255]), 'array', np.asarray([1, 0, 1])),
        (None, 'qcolor', DEFAULT_COLOR_QCOLOR),
    ],
)
def test_get_color(
    qtbot: QtBot,
    color: str | np.ndarray | None,
    mode: Literal['hex', 'array', 'qcolor'],
    expected: str | np.ndarray | QColor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the get_color utility function."""

    def _mock_exec_(self):
        """Mock exec_ method to always return Accepted."""
        qtbot.addWidget(self)
        return QColorDialog.DialogCode.Accepted

    widget = QMainWindow()
    qtbot.addWidget(widget)

    monkeypatch.setattr(QColorDialog, 'exec_', _mock_exec_)

    color_ = get_color(color, mode)
    _assert_eq(color_, expected, f'Expected color to be {expected}')


@pytest.mark.parametrize(
    ('color', 'mode'),
    [
        (None, 'hex'),
        ('#FF00FF', 'hex'),
        (None, 'array'),
        (np.asarray([255, 0, 255]), 'array'),
        (None, 'qcolor'),
    ],
)
def test_get_color_reject(
    qtbot: QtBot,
    color: str | np.ndarray | None,
    mode: Literal['hex', 'array', 'qcolor'],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the get_color utility function."""

    def _mock_exec_(self):
        """Mock exec_ method to always return Accepted."""
        qtbot.addWidget(self)
        return QColorDialog.DialogCode.Rejected

    widget = QMainWindow()
    qtbot.addWidget(widget)

    monkeypatch.setattr(QColorDialog, 'exec_', _mock_exec_)

    color_ = get_color(color, mode)
    assert color_ is None, 'Expected color to be None when dialog is rejected'
