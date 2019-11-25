from functools import reduce
from operator import ior
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget, QWidget, QHBoxLayout


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


class QMinimalTitleBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        # layout.addWidget(QPushButton())
        self.setLayout(layout)
        self.setGeometry(0, 0, 50, 50)
        self.setVisible(True)
        self.setStyleSheet("QHBoxLayout{background: yellow}")
        self.setObjectName("QMinimalTitleBar")


class QtMinimalDockWidget(QtViewerDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        title = QMinimalTitleBar()
        self.setTitleBarWidget(title)
        self.setObjectName("QtMinimalDockWidget")
