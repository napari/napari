import re
from collections import OrderedDict

from qtpy.QtCore import QEvent, QPoint, Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QItemDelegate,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...layers import Image, Labels, Points, Shapes, Surface, Vectors
from ...settings import get_settings
from ...utils.action_manager import action_manager
from ...utils.interactions import Shortcut
from ...utils.translations import trans
from ..widgets.qt_message_popup import WarnPopup

# Dict used to format strings returned from converted key press events.
# For example, the ShortcutTranslator returns 'Ctrl' instead of 'Control'.
# In order to be consistent with the code base, the values in KEY_SUBS will
# be subsituted.
KEY_SUBS = {'Ctrl': 'Control'}


class ShortcutEditor(QWidget):
    """Widget to edit keybindings for napari."""

    valueChanged = Signal(dict)
    VIEWER_KEYBINDINGS = trans._('Viewer key bindings')

    def __init__(
        self,
        parent: QWidget = None,
        description: str = "",
        value: dict = None,
    ):

        super().__init__(parent=parent)

        # Flag to not run _set_keybinding method after setting special symbols.
        # When changing line edit to special symbols, the _set_keybinding
        # method will be called again (and breaks) and is not needed.
        self._skip = False

        layers = [
            Image,
            Labels,
            Points,
            Shapes,
            Surface,
            Vectors,
        ]

        self.key_bindings_strs = OrderedDict()

        # widgets
        self.layer_combo_box = QComboBox(self)
        self._label = QLabel(self)
        self._table = QTableWidget(self)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setShowGrid(False)
        self._restore_button = QPushButton(trans._("Restore All Keybindings"))

        # Set up dictionary for layers and associated actions.
        all_actions = action_manager._actions.copy()
        self.key_bindings_strs[self.VIEWER_KEYBINDINGS] = []

        for layer in layers:
            if len(layer.class_keymap) == 0:
                actions = []
            else:
                actions = action_manager._get_layer_actions(layer)
                for name, action in actions.items():
                    all_actions.pop(name)
            self.key_bindings_strs[f"{layer.__name__} layer"] = actions

        # Left over actions can go here.
        self.key_bindings_strs[self.VIEWER_KEYBINDINGS] = all_actions

        # Widget set up
        self.layer_combo_box.addItems(list(self.key_bindings_strs))
        self.layer_combo_box.activated[str].connect(self._set_table)
        self.layer_combo_box.setCurrentText(self.VIEWER_KEYBINDINGS)
        self._set_table()
        self._label.setText(trans._("Group"))
        self._restore_button.clicked.connect(self.restore_defaults)

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

    def restore_defaults(self):
        """Launches dialog to confirm restore choice."""

        response = QMessageBox.question(
            self,
            trans._("Restore Shortcuts"),
            trans._("Are you sure you want to restore default shortcuts?"),
            QMessageBox.RestoreDefaults | QMessageBox.Cancel,
            QMessageBox.RestoreDefaults,
        )

        if response == QMessageBox.RestoreDefaults:
            self._reset_shortcuts()

    def _reset_shortcuts(self):
        """Reset shortcuts to default settings."""

        get_settings().shortcuts.reset()
        for (
            action,
            shortcuts,
        ) in get_settings().shortcuts.shortcuts.items():
            action_manager.unbind_shortcut(action)
            for shortcut in shortcuts:
                action_manager.bind_shortcut(action, shortcut)

        self._set_table(layer_str=self.layer_combo_box.currentText())

    def _set_table(self, layer_str: str = ''):
        """Builds and populates keybindings table.

        Parameters
        ----------
        layer_str : str
            If layer_str is not empty, then show the specified layers'
            keybinding shortcut table.
        """

        # Keep track of what is in each column.
        self._action_name_col = 0
        self._icon_col = 1
        self._shortcut_col = 2
        self._action_col = 3

        # Set header strings for table.
        header_strs = ['', '', '', '']
        header_strs[self._action_name_col] = trans._('Action')
        header_strs[self._shortcut_col] = trans._('Keybinding')

        # If no layer_str, then set the page to the viewer keybindings page.
        if layer_str == '':
            layer_str = self.VIEWER_KEYBINDINGS

        # If rebuilding the table, then need to disconnect the connection made
        # previously as well as clear the table contents.
        try:
            self._table.cellChanged.disconnect(self._set_keybinding)
        except TypeError:
            # if building the first time, the cells are not yet connected so this would fail.
            pass
        except RuntimeError:
            # Needed to pass some tests.
            pass

        self._table.clearContents()

        # Table styling set up.
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setStyleSheet(
            'border-bottom: 2px solid white;'
        )

        # Get all actions for the layer.
        actions = self.key_bindings_strs[layer_str]

        if len(actions) > 0:

            # Set up table based on number of actions and needed columns.
            self._table.setRowCount(len(actions))
            self._table.setColumnCount(4)
            # Set up delegate in order to capture keybindings.
            self._table.setItemDelegateForColumn(
                self._shortcut_col, ShortcutDelegate(self._table)
            )
            self._table.setHorizontalHeaderLabels(header_strs)
            self._table.verticalHeader().setVisible(False)

            # Hide the column with action names.  These are kept here for reference when needed.
            self._table.setColumnHidden(self._action_col, True)

            # Column set up.
            self._table.setColumnWidth(self._action_name_col, 250)
            self._table.setColumnWidth(self._shortcut_col, 200)
            self._table.setColumnWidth(self._icon_col, 50)

            # Go through all the actions in the layer and add them to the table.
            for row, (action_name, action) in enumerate(actions.items()):

                shortcuts = action_manager._shortcuts.get(action_name, [])
                # Set action description.  Make sure its not selectable/editable.
                item = QTableWidgetItem(action.description)
                item.setFlags(Qt.NoItemFlags)
                self._table.setItem(row, self._action_name_col, item)

                # Create empty item in order to make sure this column is not
                # selectable/editable.
                item = QTableWidgetItem("")
                item.setFlags(Qt.NoItemFlags)
                self._table.setItem(row, self._icon_col, item)

                # Set the shortcuts in table.
                item_shortcut = QTableWidgetItem(
                    Shortcut(list(shortcuts)[0]).platform if shortcuts else ""
                )
                self._table.setItem(row, self._shortcut_col, item_shortcut)

                # action_name is stored in the table to use later, but is not shown on dialog.
                item_action = QTableWidgetItem(action_name)
                self._table.setItem(row, self._action_col, item_action)

            # If a cell is changed, run .set_keybinding.
            self._table.cellChanged.connect(self._set_keybinding)
        else:
            # Display that there are no actions for this layer.
            self._table.setRowCount(1)
            self._table.setColumnCount(4)
            self._table.setHorizontalHeaderLabels(header_strs)
            self._table.verticalHeader().setVisible(False)

            self._table.setColumnHidden(self._action_col, True)
            item = QTableWidgetItem(trans._('No key bindings'))
            item.setFlags(Qt.NoItemFlags)
            self._table.setItem(0, 0, item)

    def _set_keybinding(self, row, col):
        """Checks the new keybinding to determine if it can be set.

        Parameters
        ----------
        row : int
            Row in keybindings table that is being edited.
        col : int
            Column being edited (shortcut column).
        """

        if self._skip is True:
            # Do nothing if the text is setting to a symbol.
            # Its already been handled.
            self._skip = False
            return

        if col == self._shortcut_col:
            # Get all layer actions and viewer actions in order to determine
            # the new shortcut is not already set to an action.

            current_layer_text = self.layer_combo_box.currentText()
            layer_actions = self.key_bindings_strs[current_layer_text]
            actions_all = layer_actions.copy()
            if current_layer_text is not self.VIEWER_KEYBINDINGS:
                viewer_actions = self.key_bindings_strs[
                    self.VIEWER_KEYBINDINGS
                ]

                actions_all.update(viewer_actions)

            # get the current item from shortcuts column
            current_item = self._table.currentItem()
            new_shortcut = current_item.text()
            new_shortcut = new_shortcut[0].upper() + new_shortcut[1:]

            # get the current action name
            current_action = self._table.item(row, self._action_col).text()

            # get the original shortcutS
            current_shortcuts = list(
                action_manager._shortcuts.get(current_action, {})
            )

            # Flag to indicate whether to set the new shortcut.
            replace = True
            # Go through all layer actions to determine if the new shortcut is already here.
            for row1, (action_name, action) in enumerate(actions_all.items()):
                shortcuts = action_manager._shortcuts.get(action_name, [])

                if new_shortcut in shortcuts:
                    # Shortcut is here (either same action or not), don't replace in settings.
                    replace = False
                    if action_name != current_action:
                        # the shortcut is saved to a different action

                        # show warning symbols
                        self._show_warning_icons([row, row1])

                        # show warning message
                        message = trans._(
                            "The keybinding <b>{new_shortcut}</b>  is already assigned to <b>{action_description}</b>; change or clear that shortcut before assigning <b>{new_shortcut}</b> to this one.",
                            new_shortcut=new_shortcut,
                            action_description=action.description,
                        )
                        self._show_warning(new_shortcut, action, row, message)

                        if len(current_shortcuts) > 0:
                            # If there was a shortcut set originally, then format it and reset the text.
                            format_shortcut = Shortcut(
                                current_shortcuts[0]
                            ).platform

                            if format_shortcut != current_shortcuts[0]:
                                # only skip the next round if there are special symbols
                                self._skip = True

                            current_item.setText(format_shortcut)

                        else:
                            # There wasn't a shortcut here.
                            current_item.setText("")

                        self._cleanup_warning_icons([row, row1])

                        break

                    else:
                        # This shortcut was here.  Reformat and reset text.
                        format_shortcut = Shortcut(new_shortcut).platform
                        if format_shortcut != new_shortcut:
                            # Only skip the next round if there are special symbols in shortcut.
                            self._skip = True

                        current_item.setText(format_shortcut)

            if replace is True:
                # This shortcut is not taken.

                #  Unbind current action from shortcuts in action manager.
                action_manager.unbind_shortcut(current_action)

                if new_shortcut != "":
                    # Bind the new shortcut.
                    try:
                        action_manager.bind_shortcut(
                            current_action, new_shortcut
                        )
                    except TypeError:
                        # Shortcut is not valid.
                        action_manager._shortcuts[current_action] = set()
                        # need to rebind the old shortcut
                        action_manager.unbind_shortcut(current_action)
                        action_manager.bind_shortcut(
                            current_action, current_shortcuts[0]
                        )

                        # Show warning message to let user know this shortcut is invalid.
                        self._show_warning_icons([row])

                        message = trans._(
                            "<b>{new_shortcut}</b> is not a valid keybinding.",
                            new_shortcut=new_shortcut,
                        )
                        self._show_warning(new_shortcut, action, row, message)

                        self._cleanup_warning_icons([row])

                        format_shortcut = Shortcut(
                            current_shortcuts[0]
                        ).platform
                        if format_shortcut != current_shortcuts[0]:
                            # Skip the next round if there are special symbols.
                            self._skip = True

                        # Update text to formated shortcut.
                        current_item.setText(format_shortcut)
                        return

                    # The new shortcut is valid and can be displayed in widget.

                    # Keep track of what changed.
                    new_value_dict = {current_action: [new_shortcut]}

                    # Format new shortcut.
                    format_shortcut = Shortcut(new_shortcut).platform
                    if format_shortcut != new_shortcut:
                        # Skip the next round because there are special symbols.
                        self._skip = True

                    # Update text to formated shortcut.
                    current_item.setText(format_shortcut)

                else:
                    # There is not a new shortcut to bind.  Keep track of it.
                    if action_manager._shortcuts[current_action] != "":
                        new_value_dict = {current_action: [""]}

                if new_value_dict:
                    # Emit signal when new value set for shortcut.
                    self.valueChanged.emit(new_value_dict)

    def _show_warning_icons(self, rows):
        """Creates and displays the warning icons.

        Parameters
        ----------
        rows : list[int]
            List of row numbers that should have the icon.
        """

        for row in rows:
            self.warning_indicator = QLabel(self)
            self.warning_indicator.setObjectName("error_label")

            self._table.setCellWidget(
                row, self._icon_col, self.warning_indicator
            )

    def _cleanup_warning_icons(self, rows):
        """Remove the warning icons from the shortcut table.

        Parameters
        ----------
        rows : list[int]
            List of row numbers to remove warning icon from.

        """
        for row in rows:
            self._table.setCellWidget(row, self._icon_col, QLabel(""))

    def _show_warning(self, new_shortcut='', action=None, row=0, message=''):
        """Creates and displays warning message when shortcut is already assigned.

        Parameters
        ----------
        new_shortcut : str
            The new shortcut attempting to be set.
        action : Action
            Action that is already assigned with the shortcut.
        row : int
            Row in table where the shortcut is attempting to be set.
        message : str
            Message to be displayed in warning pop up.
        """

        # Determine placement of warning message.
        delta_y = 105
        delta_x = 10
        global_point = self.mapToGlobal(
            QPoint(
                self._table.columnViewportPosition(self._shortcut_col)
                + delta_x,
                self._table.rowViewportPosition(row) + delta_y,
            )
        )

        # Create warning pop up and move it to desired position.
        self._warn_dialog = WarnPopup(
            text=message,
        )
        self._warn_dialog.move(global_point)

        # Styling adjustments.
        self._warn_dialog.resize(250, self._warn_dialog.sizeHint().height())

        self._warn_dialog._message.resize(
            200, self._warn_dialog._message.sizeHint().height()
        )

        self._warn_dialog.exec_()

    def value(self):
        """Return the actions and shortcuts currently assigned in action manager.

        Returns
        -------
        value: dict
            Dictionary of action names and shortcuts assigned to them.
        """

        value = {}

        for row, (action_name, action) in enumerate(
            action_manager._actions.items()
        ):
            shortcuts = action_manager._shortcuts.get(action_name, [])
            value[action_name] = list(shortcuts)

        return value


class ShortcutDelegate(QItemDelegate):
    """Delegate that handles when user types in new shortcut."""

    def createEditor(self, QWidget, QStyleOptionViewItem, QModelIndex):
        self._editor = EditorWidget(QWidget)
        return self._editor

    def setEditorData(self, widget, QModelIndex):
        text = QModelIndex.model().data(QModelIndex, Qt.EditRole)
        widget.setText(str(text) if text else "")

    def updateEditorGeometry(self, QWidget, QStyleOptionViewItem, QModelIndex):
        QWidget.setGeometry(QStyleOptionViewItem.rect)

    def setModelData(self, widget, QAbstractItemModel, QModelIndex):
        text = widget.text()
        QAbstractItemModel.setData(QModelIndex, text, Qt.EditRole)


class EditorWidget(QLineEdit):
    """Editor widget set in the delegate column in shortcut table."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def event(self, event):
        """Qt method override."""
        if event.type() == QEvent.ShortcutOverride:
            self.keyPressEvent(event)
            return True
        elif event.type() in [QEvent.KeyPress, QEvent.Shortcut]:
            return True
        else:
            return super().event(event)

    def keyPressEvent(self, event):
        """Qt method override."""
        event_key = event.key()
        if not event_key or event_key == Qt.Key_unknown:
            return

        if event_key in [
            Qt.Key_Control,
            Qt.Key_Shift,
            Qt.Key_Alt,
            Qt.Key_Meta,
            Qt.Key_Return,
            Qt.Key_Tab,
            Qt.Key_CapsLock,
        ]:
            # Do not allow user to set these keys as shortcut.
            return

        # Translate key value to key string.
        translator = ShortcutTranslator()
        event_keyseq = translator.keyevent_to_keyseq(event)
        event_keystr = event_keyseq.toString(QKeySequence.PortableText)

        # Split the shortcut if it contains a symbol.
        parsed = re.split('[-(?=.+)]', event_keystr)

        keys = []
        # Format how the shortcut is written (ex. 'Ctrl+B' is changed to 'Control-B')
        for val in parsed:
            if val in KEY_SUBS.keys():
                keys.append(KEY_SUBS[val])
            else:
                keys.append(val)

        keys = '-'.join(keys)
        self.setText(keys)


class ShortcutTranslator(QKeySequenceEdit):
    """
    Convert QKeyEvent into QKeySequence.
    """

    def __init__(self):
        super().__init__()
        self.hide()

    def keyevent_to_keyseq(self, event):
        """Return a QKeySequence representation of the provided QKeyEvent."""
        self.keyPressEvent(event)
        event.accept()
        return self.keySequence()

    def keyReleaseEvent(self, event):
        """Qt Override"""
        return False

    def timerEvent(self, event):
        """Qt Override"""
        return False

    def event(self, event):
        """Qt Override"""
        return False
