"""Status bar widget on the viewer MainWindow"""
from typing import TYPE_CHECKING, Optional, cast

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStatusBar,
    QWidget,
)
from superqt import QElidingLabel

from napari._qt.dialogs.qt_activity_dialog import ActivityToggleItem
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import _QtMainWindow


class ViewerStatusBar(QStatusBar):
    def __init__(self, parent: '_QtMainWindow') -> None:
        super().__init__(parent=parent)

        main_widget = QWidget()

        layout = QHBoxLayout()

        self._status = QLabel(trans._('Ready'))
        self._status.setContentsMargins(0, 0, 0, 0)

        self._layer_base = QElidingLabel(trans._(''))
        self._layer_base.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._layer_base.setMinimumSize(100, 16)
        self._layer_base.setContentsMargins(0, 0, 0, 0)
        self._layer_base.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Maximum
        )

        self._plugin_reader = QElidingLabel(trans._(''))
        self._plugin_reader.setMinimumSize(80, 16)
        self._plugin_reader.setContentsMargins(0, 0, 0, 0)
        self._plugin_reader.setElideMode(Qt.TextElideMode.ElideMiddle)

        self._source_type = QLabel('')
        self._source_type.setContentsMargins(0, 0, 0, 0)

        self._coordinates = QLabel('')
        self._coordinates.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._status)
        layout.addWidget(self._layer_base)
        layout.addWidget(self._source_type)
        layout.addWidget(self._plugin_reader)
        layout.addWidget(self._coordinates)
        layout.addStretch(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        main_widget.setLayout(layout)

        self.addWidget(main_widget, 1)
        self._help = QElidingLabel('')
        self._help.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._help.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addWidget(self._help, 1)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        # FIXME: feels weird to set this here.
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)

    def setHelpText(self, text: str) -> None:
        self._help.setText(text)

    def setStatusText(
        self,
        text: str = "",
        layer_base: str = "",
        source_type=None,
        plugin: str = "",
        coordinates: str = "",
    ) -> None:
        # The method used to set a single value as the status and not
        # all the layer information.

        self._status.setText(text)

        self._layer_base.setVisible(bool(layer_base))
        self._layer_base.setText(layer_base)

        self._source_type.setVisible(bool(source_type))
        if source_type:
            self._source_type.setText(f'{source_type}: ')

        self._plugin_reader.setVisible(bool(plugin))

        self._plugin_reader.setText(plugin)

        self._coordinates.setVisible(bool(coordinates))
        self._coordinates.setText(coordinates)

    def _toggle_activity_dock(self, visible: Optional[bool] = None):
        par = cast(_QtMainWindow, self.parent())
        if visible is None:
            visible = not par._activity_dialog.isVisible()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(
                Qt.ArrowType.DownArrow
            )
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.ArrowType.UpArrow)
