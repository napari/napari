from functools import reduce
from operator import ior
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QWidget,
    QHBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
)


class QtViewerDockWidget(QDockWidget):
    """Wrap a QWidget in a QDockWidget and forward viewer events

    Parameters
    ----------
    widget : QWidget
        `widget` that will be added as QDockWidget's main widget.
    name : str
        Name of dock widget.
    area : str
        Side of the main window to which the new dock widget will be added.
        Must be in {'left', 'right', 'top', 'bottom'}
    allowed_areas : list[str], optional
        Areas, relative to main window, that the widget is allowed dock.
        Each item in list must be in {'left', 'right', 'top', 'bottom'}
        By default, all areas are allowed.
    """

    def __init__(
        self,
        viewer,
        widget: QWidget,
        *,
        name: str = '',
        area: str = 'bottom',
        allowed_areas: Optional[List[str]] = None,
    ):
        self.viewer = viewer
        super().__init__(name)
        self.name = name

        areas = {
            'left': Qt.LeftDockWidgetArea,
            'right': Qt.RightDockWidgetArea,
            'top': Qt.TopDockWidgetArea,
            'bottom': Qt.BottomDockWidgetArea,
        }
        if area not in areas:
            raise ValueError(f'area argument must be in {list(areas.keys())}')
        self.area = area
        self.qt_area = areas[area]

        if allowed_areas:
            if not isinstance(allowed_areas, (list, tuple)):
                raise TypeError('`allowed_areas` must be a list or tuple')
            if not all(area in areas for area in allowed_areas):
                raise ValueError(
                    f'all allowed_areas argument must be in {list(areas.keys())}'
                )
            allowed_areas = reduce(ior, [areas[a] for a in allowed_areas])
        else:
            allowed_areas = Qt.AllDockWidgetAreas
        self.setAllowedAreas(allowed_areas)
        self.setMinimumHeight(50)
        self.setMinimumWidth(50)
        self.setObjectName(name)

        self.setWidget(widget)
        widget.setParent(self)
        self._features = self.features()
        self.dockLocationChanged.connect(self._set_title_orientation)

    def setFeatures(self, features):
        super().setFeatures(features)
        self._features = self.features()

    def keyPressEvent(self, event):
        # if you subclass QtViewerDockWidget and override the keyPressEvent
        # method, be sure to call super().keyPressEvent(event) at the end of
        # your method to pass uncaught key-combinations to the viewer.
        return self.viewer.keyPressEvent(event)

    def _set_title_orientation(self, area):
        if area in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea):
            features = self._features
        else:
            features = self._features | self.DockWidgetVerticalTitleBar
        self.setFeatures(features)


class QMinimalTitleBar(QLabel):
    """A widget to be used as the titleBar in the QtMinimalDock Widget.

    Keeps vertical size minimal, has a hand cursor and styles (in stylesheet)
    for hover.
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("QMinimalTitleBar")

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(4)

        line = QFrame(self)
        line.setFixedHeight(1)
        line.setObjectName("QMinimalTitleBarLine")

        self.close_button = QPushButton(self)
        self.close_button.setCursor(Qt.ArrowCursor)

        layout.addWidget(self.close_button)
        layout.addWidget(line)

        self.setLayout(layout)
        self.setCursor(Qt.OpenHandCursor)

    def sizeHint(self):
        # this seems to be the correct way to set the height of the titlebar
        szh = super().sizeHint()
        szh.setHeight(18)
        return szh


class QtMinimalDockWidget(QtViewerDockWidget):
    """A subclass that has a small but visible titlebar for floating and moving
    the widget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("QtMinimalDockWidget")

        self.title = QMinimalTitleBar()
        self.setTitleBarWidget(self.title)

        self.toggle_visibility = self.toggleViewAction().trigger
        self.title.close_button.clicked.connect(self.toggle_visibility)
        # self.topLevelChanged.connect(self._on_top_level_change)

    def _on_top_level_change(self, event):
        # if connected, this will give a native title bar to floated windows...
        # however, I haven't yet been able to prevent the "permanent-floating"
        # problem once a floated window is closed with the native button.
        # so this is currently unconnected
        if self.isFloating():
            self.setTitleBarWidget(None)
        else:
            self.setTitleBarWidget(self.title)
