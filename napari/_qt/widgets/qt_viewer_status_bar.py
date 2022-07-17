"""Status bar widget on the viewer MainWindow"""
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStatusBar,
    QWidget,
)
from superqt import QElidingLabel

from ...utils.translations import trans
from ..dialogs.qt_activity_dialog import ActivityToggleItem

if TYPE_CHECKING:
    from ..qt_main_window import _QtMainWindow


class ViewerStatusBar(QStatusBar):
    def __init__(self, parent: '_QtMainWindow') -> None:
        super().__init__(parent=parent)

        main_widget = QWidget()

        layout = QHBoxLayout()

        self._status = QLabel('Ready')
        self._status.setMinimumSize(0, 16)
        self._status.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        self._layer_base = QElidingLabel(trans._(''))
        self._layer_base.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._layer_base.setMinimumSize(80, 16)
        self._layer_base.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Maximum
        )

        self._plugin_reader = QElidingLabel(trans._(''))
        self._plugin_reader.setMinimumSize(80, 16)
        self._plugin_reader.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Maximum
        )

        self._plugin_reader.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._source_type = QLabel('')
        self._source_type.setMinimumSize(50, 16)
        self._source_type.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Maximum
        )

        self._coordinates = QLabel('')
        self._coordinates.setMinimumSize(80, 16)
        self._coordinates.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Maximum
        )

        layout.addWidget(self._status)
        layout.addWidget(self._layer_base)
        layout.addWidget(self._source_type)
        layout.addWidget(self._plugin_reader)
        layout.addWidget(self._coordinates)

        main_widget.setLayout(layout)

        self.addWidget(main_widget)
        self._help = QLabel('')
        self.addPermanentWidget(self._help)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        # FIXME: feels weird to set this here.
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def setHelpText(self, text: str) -> None:
        self._help.setText(text)

    def setStatusText(
        self,
        text: str = None,
        layer_base=None,
        source_type=None,
        plugin=None,
        coordinates=None,
    ) -> None:
        # The method used to set a single value as the status and not
        # all the layer information.

        if text:
            self._status.setText(text)
        else:
            self._status.setText(' ')

        if layer_base:
            self._layer_base.show()
            self._layer_base.setText(layer_base)
        else:
            self._layer_base.hide()

        if source_type:
            self._source_type.show()
            self._source_type.setText(f'{source_type}: ')
        else:
            self._source_type.hide()

        if plugin:
            self._plugin_reader.show()
            self._plugin_reader.setText(plugin)
        else:
            self._plugin_reader.hide()

        if coordinates:
            self._coordinates.show()
            self._coordinates.setText(coordinates)
        else:
            self._coordinates.hide()

    def _toggle_activity_dock(self, visible: bool):
        par: _QtMainWindow = self.parent()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(
                Qt.ArrowType.DownArrow
            )
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.ArrowType.UpArrow)
