"""Status bar widget on the viewer MainWindow"""
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget
from superqt import QElidingLabel

from ...utils.translations import trans
from ..dialogs.qt_activity_dialog import ActivityToggleItem

if TYPE_CHECKING:
    from ..qt_main_window import _QtMainWindow

STATUS_FRACTION_WIDTH = 0.7


class ViewerStatusBar(QStatusBar):
    def __init__(self, parent: '_QtMainWindow') -> None:
        super().__init__(parent=parent)

        main_widget = QWidget()

        layout = QHBoxLayout()

        self._layer_base = QElidingLabel(trans._('Ready 1'))
        self._layer_base.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._plugin_reader = QElidingLabel(trans._('Ready 2'))
        self._plugin_reader.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._source_type = QLabel('ready 3')
        self._coordinates = QLabel('ready 4')

        layout.addWidget(self._layer_base)
        layout.addWidget(QLabel('source: '))
        layout.addWidget(self._plugin_reader)
        layout.addWidget(self._source_type)
        layout.addWidget(self._coordinates)

        main_widget.setLayout(layout)

        # self.source.reader_plugin

        # self._status_message = QElidingLabel(trans._('Ready'))
        # self._status_message.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.addWidget(main_widget)
        # self.addWidget(self._status_message)
        self._help = QLabel('')
        self.addPermanentWidget(self._help)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        # FIXME: feels weird to set this here.
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)

    def setHelpText(self, text: str) -> None:
        self._help.setText(text)

    def setStatusText(self, text: str) -> None:

        width = int(self.parent().width() * STATUS_FRACTION_WIDTH)

        idx1 = text.find(':') + 3
        idx2 = text.find(',') - 1
        layer_base = text[idx1:idx2]
        if layer_base != self._layer_base.text():
            self._layer_base.resize(width, self._layer_base.height())
            self._layer_base.setText(layer_base)

        text = text[idx2 + 3 :]
        idx1 = text.find(':') + 3
        idx2 = text.find(',') - 1
        source_type = text[idx1:idx2]
        if source_type != self._source_type.text():
            self._source_type.setText(f'({source_type})')

        text = text[idx2 + 3 :]
        idx1 = text.find(':') + 3
        idx2 = text.find('}') - 1
        reader = text[idx1:idx2]
        if reader != self._plugin_reader.text():
            self._plugin_reader.setText(reader)

        text = text[idx2 + 3 :]
        idx1 = text.find('[')
        coordinates = text[idx1:]
        self._coordinates.setText(coordinates)

    def _toggle_activity_dock(self, visible: bool):
        par: _QtMainWindow = self.parent()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(Qt.DownArrow)
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.UpArrow)
