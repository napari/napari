from typing import Any, TypeVar

import pytest
from pytestqt.qtbot import QtBot

from napari._qt.layer_controls.dynamic.widgets import QtWidgetControlsBase

T = TypeVar('T', bound=QtWidgetControlsBase)


class QtWrap:
    def __init__(self, qtbot: QtBot) -> None:
        self._qtbot = qtbot
        self._controls = []

    def add_control(self, control: T) -> T:
        self._controls.append(control)
        for label, widget in control.get_widget_controls():
            self._qtbot.add_widget(widget)
            self._qtbot.add_widget(label)
        return control

    def clear(self):
        for control in self._controls:
            control.deleteLater()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._qtbot, name)


@pytest.fixture
def qt_wrap(qtbot):
    q = QtWrap(qtbot)
    yield q
    q.clear()
