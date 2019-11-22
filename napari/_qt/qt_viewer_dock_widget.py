from qtpy.QtWidgets import QDockWidget, QWidget
from qtpy.QtCore import Qt


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
    allowed_areas : Qt.DockWidgetArea, optional
        Areas, relative to main window, that the new dock is allowed to go.
    """

    def __init__(
        self,
        viewer,
        widget: QWidget,
        *,
        name: str = '',
        area: str = 'bottom',
        allowed_areas=None,
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
            raise ValueError(f'side argument must be in {list(areas.keys())}')
        self.area = area
        self.qt_area = areas[area]

        self.setAllowedAreas(
            allowed_areas
            or (
                Qt.LeftDockWidgetArea
                | Qt.BottomDockWidgetArea
                | Qt.RightDockWidgetArea
                | Qt.TopDockWidgetArea
            )
        )
        self.setMinimumHeight(50)
        self.setMinimumWidth(50)
        self.setObjectName(name)

        self.setWidget(widget)
        widget.setParent(self)

    def keyPressEvent(self, event):
        return self.viewer.keyPressEvent(event)
