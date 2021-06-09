from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...utils.action_manager import action_manager
from ...utils.settings import SETTINGS
from ...utils.translations import trans

# from qtpy.QtCore import Qt.ItemIsEditable


class ShortcutEditor(QDialog):
    """ """

    def __init__(
        self,
        parent: QWidget = None,
        description: str = "",
        value: dict = None,
    ):

        super().__init__(parent)

        # widgets
        self.layer_combo_box = QComboBox(self)
        self._label = QLabel(self)
        self._table = QTableWidget(self)

        # Set up buttons
        self._restore_button = QPushButton(trans._("Reset All Keybindings"))

        # set up
        self.layer_combo_box.addItems(['All Active Keybindings'])
        self._label.setText("Layer")

        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._table.setRowCount(len(SETTINGS.shortcuts.shortcuts))
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(['Action', 'Keybinding'])
        self._table.verticalHeader().setVisible(False)
        # myTableWidget->verticalHeader()->setVisible(false);

        for row, (action_name, action) in enumerate(
            action_manager._actions.items()
        ):
            shortcuts = action_manager._shortcuts.get(action_name, [])
            # for row, (action_name, shortcuts) in enumerate(SETTINGS.shortcuts.shortcuts.items()):
            # shortcuts = action_manager._shortcuts.get(action_name, [])
            item = QTableWidgetItem(action_name)
            item = QTableWidgetItem(action.description)

            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            self._table.setItem(row, 0, item)
            item_shortcut = QTableWidgetItem(
                list(shortcuts)[0] if shortcuts else ""
            )

            self._table.setItem(row, 1, item_shortcut)

        self._table.itemChanged.connect(self._set_keybinding)
        self._table.cellChanged.connect(self._set_keybinding2)

        # layout

        hlayout1 = QHBoxLayout()

        hlayout1.addWidget(self._label)
        hlayout1.addWidget(self.layer_combo_box)
        hlayout1.setContentsMargins(0, 0, 0, 0)
        hlayout1.setSpacing(20)
        hlayout1.addStretch(0)

        hlayout2 = QHBoxLayout()
        hlayout2.addLayout(hlayout1)
        hlayout2.addWidget(self._restore_button)

        layout = QVBoxLayout()
        layout.addLayout(hlayout2)
        layout.addWidget(self._table)

        self.setLayout(layout)

    def _set_keybinding(self, event):
        # print(event)
        print(event.text())

    def _set_keybinding2(self, event):
        # event is the index of the event.

        # get the current item (keyboard shortcut)
        shortcut = self._table.currentItem()
        # get the current action for the keyboard shortcut
        item2 = self._table.item(event, 0).text()
        print(item2)
        # print(action_manager._actions.items()[event]._description)
        # print(event.text())

        # check if shortcut is in use in layer or globally
        print(action_manager._shortcuts)
        for row, (action_name, action) in enumerate(
            action_manager._actions.items()
        ):
            shortcuts = action_manager._shortcuts.get(action_name, [])

            if shortcut in shortcuts:
                # this is here, give warning
                # pop up window for warning.
                pass
