from collections import OrderedDict

from qtpy.QtCore import Qt  # Signal
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

from ...layers import Image, Labels, Points, Shapes, Surface, Vectors
from ...utils.action_manager import action_manager
from ...utils.settings import SETTINGS
from ...utils.translations import trans

# from qtpy.QtCore import Qt.ItemIsEditable


class ShortcutEditor(QDialog):
    """ """

    # onChanged = Signal(list[str, str])

    def __init__(
        self,
        # viewer,
        # key_map_handler,
        parent: QWidget = None,
        description: str = "",
        value: dict = None,
    ):

        super().__init__(parent=parent)

        #  for name, shortcut in self._shortcuts.items():
        #         action = self._actions.get(name, None)
        #         if action and layer == action.keymapprovider:
        #             layer_shortcuts[layer][str(shortcut)] = action.description

        # col = theme['secondary']
        layers = [
            Image,
            Labels,
            Points,
            Shapes,
            Surface,
            Vectors,
        ]

        self.key_bindings_strs = OrderedDict()
        self.key_bindings_strs[trans._('All active key bindings')] = 'base'

        # widgets
        self.layer_combo_box = QComboBox(self)
        self._label = QLabel(self)
        self._table = QTableWidget(self)

        # Set up buttons
        self._restore_button = QPushButton(trans._("Reset All Keybindings"))

        # set up
        for layer in layers:
            self.key_bindings_strs[f"{layer.__name__} layer"] = layer

        self.layer_combo_box.addItems(list(self.key_bindings_strs))
        self._label.setText("Layer")

        # layer_shortcuts = action_manager._get_layer_shortcuts(layers)
        # print(layer_shortcuts)

        #     if len(layer.class_keymap) == 0:
        #         text = trans._('No key bindings')
        #     else:
        #         text = get_key_bindings_summary(
        #             layer_shortcuts[layer], col=col
        #         )

        self._set_table()

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

    def _set_table(self, layer=None):

        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._table.setRowCount(len(action_manager._actions))
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(['Action', 'Keybinding'])
        self._table.verticalHeader().setVisible(False)

        self._table.setColumnHidden(2, True)

        for row, (action_name, action) in enumerate(
            action_manager._actions.items()
        ):
            shortcuts = action_manager._shortcuts.get(action_name, [])
            # for row, (action_name, shortcuts) in enumerate(SETTINGS.shortcuts.shortcuts.items()):
            # shortcuts = action_manager._shortcuts.get(action_name, [])
            item = QTableWidgetItem(action.description)

            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            self._table.setItem(row, 0, item)
            item_shortcut = QTableWidgetItem(
                list(shortcuts)[0] if shortcuts else ""
            )

            self._table.setItem(row, 1, item_shortcut)

            item_action = QTableWidgetItem(action_name)
            # action_name is stored in the table to use later, but is not shown on dialog.
            self._table.setItem(row, 2, item_action)

        # self._table.itemChanged.connect(self._set_keybinding)
        self._table.cellChanged.connect(self._set_keybinding)

    def _set_keybinding(self, event):
        # event is the index of the event.

        # get the current item from shortcuts column
        item1 = self._table.currentItem()
        new_shortcut = item1.text()
        # get the current action description
        # action_description = self._table.item(event, 0).text()

        # get the action name
        current_action = self._table.item(event, 2).text()
        # print(action_manager._actions.items()[event]._description)
        # print(event.text())
        #
        # check if shortcut is in use in layer or globally
        # current_shortcut = list(
        #     action_manager._shortcuts.get(current_action, [])
        # )[0]

        replace = True
        for row, (action_name, action) in enumerate(
            action_manager._actions.items()
        ):
            shortcuts = action_manager._shortcuts.get(action_name, [])

            if new_shortcut in shortcuts:
                # shortcut is here (either same action or not), don't replace in settings.
                replace = False
                if action_name != current_action:
                    # the shortcut is saved to a different action, show message.
                    # pop up window for warning.
                    message = trans._(
                        f"The keybinding <b>{new_shortcut}</b> "
                        + f"is already assigned to <b>{action.description}</b>; change or clear "
                        + f"that shortcut before assigning <b>{new_shortcut}</b> to this one."
                    )

                    print(message)

                    self._warn_dialog = KeyBindWarnPopup(
                        parent=self,
                        text=message,
                    )
                    # self._reset_dialog.valueChanged.connect(print('works!'))
                    self._warn_dialog.exec_()

                    # get the original shortcut
                    current_shortcuts = list(
                        action_manager._shortcuts.get(current_action, {})
                    )
                    # reset value in table to value stored in action manager.
                    item1.setText(current_shortcuts[0])

                    print('done!')

                    break

            # else:

        print('replace', replace)
        if replace is True:

            # this shortcut is not taken, can set it and save in settings
            # how are we doing this? saving in settings and then that triggers to update the action manager?
            # right now I'm updating the action manager, and settings will be saved after a trigger as with
            # all the other widgets.

            #  Bind new shortcut to the action manager
            print('unbind', current_action)
            action_manager.unbind_shortcut(current_action)
            print('bind', new_shortcut)
            action_manager.bind_shortcut(current_action, new_shortcut)

            #     break
            # Save to settings here temporarily
            if current_action in SETTINGS.shortcuts.shortcuts:
                # This one was here, need to save the new shortcut
                SETTINGS.shortcuts.shortcuts[current_action][0] = new_shortcut
            else:
                # this action not currently set in SETTINGS, need to save it.
                SETTINGS.shortcuts.shortcuts[current_action] = [new_shortcut]

                # self.onChanged.emit([action, new_shortcut])

    def value(self):
        # need to return value from widget.
        pass


class KeyBindWarnPopup(QDialog):
    """Dialog to inform user that shortcut is already assigned."""

    # valueChanged = Signal()

    def __init__(
        self,
        parent: QWidget = None,
        text: str = "",
    ):
        super().__init__(parent)

        # self.setWindowFlags(Qt.FramelessWindowHint)

        # Set up components
        self._message = QLabel(self)

        # Widget set up
        self._message.setText(text)
        self._message.setWordWrap(True)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self._message)

        self.setLayout(main_layout)

        # # Signals
        # self._button_cancel.clicked.connect(self.on_click_cancel)
        # self._button_restore.clicked.connect(self.on_click_restore)

    #     QDialog *popup = new QDialog(0, Qt::Popup | Qt::FramelessWindowHint);
    # QVBoxLayout *layout = new QVBoxLayout;
    # QLabel *popupLabel = new QLabel(data_.value(rowIndex).displayHtml(), 0);
    # layout->addWidget(popupLabel);
    # popupLabel->setTextFormat(Qt::RichText);
    # popupLabel->setOpenExternalLinks(true);
    # popup->setLayout(layout);
    # popup->move(location);
    # popup->exec();
