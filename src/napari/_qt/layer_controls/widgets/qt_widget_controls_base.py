from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QLabel, QWidget

from napari.layers.base.base import Layer
from napari.utils.events import disconnect_events

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    _QtABCMeta = ABCMeta

else:

    class _QtABCMeta(type(QObject), ABCMeta):
        pass


class QtWrappedLabel(QLabel):
    """
    QLabel subclass with the `wordWrap` activated (True) and text aligned
    to the right and vertically centered by default.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)
        self.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )


class QtWidgetControlsBase(QObject, metaclass=_QtABCMeta):
    """
    Base class that defines base methods for wrapper classes that do the
    connection of events/signals between layer attributes and Qt widgets.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Layer]
        A list of napari layers.
    """

    def __init__(self, parent: QWidget, layers: list[Layer]) -> None:
        super().__init__(parent)
        # Setup layer
        self._layers = layers
        # Track registered callbacks (defined via `attr_to_settr` for example)
        # so it is possible to disconnect them when the widget is being closed/deleted.
        # Arguments of callbacks are hard to track; Any is the best we can do here.
        self._callbacks: list[Callable[[Any], None]] = []

    @abstractmethod
    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        """
        Enable access to the created labels and control widgets.

        Returns
        -------
        list : list[tuple[QtWrappedLabel, QWidget]]
            List of tuples of the label and widget controls available.

        """
        raise NotImplementedError

    def disconnect_widget_controls(self) -> None:
        """
        Disconnect layers from widget controls.
        """
        for layer in self._layers:
            disconnect_events(layer.events, self)
            for callback in self._callbacks:
                disconnect_events(layer.events, callback)

    def deleteLater(self) -> None:
        self.disconnect_widget_controls()
        super().deleteLater()
