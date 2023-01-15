from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterator

from qtpy import QtCore, QtGui
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt, Signal

if TYPE_CHECKING:
    from napari._qt.qt_main_window import _QtMainWindow
    from napari.components import ViewerModel
    from napari.components.command_palette import Command


class QCommandPalette(QtW.QWidget):
    """A Qt command palette widget."""

    hidden = Signal()

    def __init__(self, parent: _QtMainWindow):
        super().__init__(parent)

        self._line = QCommandLineEdit()
        self._list = QCommandList(parent._qt_viewer.viewer)
        _layout = QtW.QVBoxLayout(self)
        _layout.addWidget(self._line)
        _layout.addWidget(self._list)
        self.setLayout(_layout)

        self._line.textChanged.connect(self._on_text_changed)
        self._list.commandClicked.connect(self._on_command_clicked)
        self._line.editingFinished.connect(self.hide)
        font = self.font()
        font.setPointSize(int(font.pointSize() * 1.2))
        self.setFont(font)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 400)

    def match_color(self) -> str:
        """The color used for the matched characters."""
        return self._list.match_color()

    def set_match_color(self, color: str):
        """Set the color used for the matched characters."""
        return self._list.set_match_color(color)

    def add_command(self, cmd: Command):
        self._list.add_command(cmd)
        return None

    def extend_command(self, list_of_commands: list[Command]):
        self._list.extend_command(list_of_commands)
        return None

    def clear_commands(self):
        self._list.clear_commands()
        return None

    def install_to(self, parent: QtW.QWidget):
        self.setParent(parent, Qt.WindowType.SubWindow)
        self.hide()

    def _on_text_changed(self, text: str):
        self._list.update_for_text(text)
        return None

    def _on_command_clicked(self, index: int):
        self._list.execute(index)
        self.hide()
        return None

    def focusOutEvent(self, a0: QtGui.QFocusEvent) -> None:
        self.hide()
        return super().focusOutEvent(a0)

    def show(self):
        self._line.setText("")
        self._list.update_for_text("")
        super().show()
        if parent := self.parentWidget():
            parent_rect = parent.rect()
            self_size = self.sizeHint()
            w = min(int(parent_rect.width() * 0.8), self_size.width())
            topleft = parent.rect().topLeft()
            topleft.setX(int(topleft.x() + (parent_rect.width() - w) / 2))
            topleft.setY(int(topleft.y() + 3))
            self.move(topleft)
            self.resize(w, self_size.height())

        self.raise_()
        self._line.setFocus()
        return None

    def hide(self):
        self.hidden.emit()
        return super().hide()

    def text(self) -> str:
        """Return the text in the line edit."""
        return self._line.text()

    def setText(self, text: str):
        """Set the text in the line edit."""
        self._line.setText(text)
        return None


class QCommandLineEdit(QtW.QLineEdit):
    """The line edit used in command palette widget."""

    def commandPalette(self) -> QCommandPalette:
        """The parent command palette widget."""
        return self.parent()

    def event(self, e: QtCore.QEvent):
        if e.type() != QtCore.QEvent.Type.KeyPress:
            return super().event(e)
        e = QtGui.QKeyEvent(e)
        if e.modifiers() in (
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.KeypadModifier,
        ):
            key = e.key()
            if key == Qt.Key.Key_Escape:
                self.commandPalette().hide()
                return True
            elif key == Qt.Key.Key_Return:
                palette = self.commandPalette()
                if palette._list.can_execute():
                    self.commandPalette().hide()
                    self.commandPalette()._list.execute()
                    return True
                else:
                    return False
            elif key == Qt.Key.Key_Up:
                self.commandPalette()._list.move_selection(-1)
                return True
            elif key == Qt.Key.Key_Down:
                self.commandPalette()._list.move_selection(1)
                return True
        return super().event(e)


def bold_colored(text: str, color: str) -> str:
    """Return a bolded and colored HTML text."""
    return f"<b><font color={color!r}>{text}</font></b>"


def colored(text: str, color: str) -> str:
    """Return a colored HTML text."""
    return f"<font color={color!r}>{text}</font>"


class QCommandMatchModel(QtCore.QAbstractListModel):
    """A list model for the command palette."""

    def __init__(self, parent: QtW.QWidget = None):
        super().__init__(parent)
        self._commands: list[Command] = []
        self._max_matches = 80

    def rowCount(self, parent: QtCore.QModelIndex = None) -> int:
        return self._max_matches

    def data(self, index: QtCore.QModelIndex, role: int = ...) -> Any:
        """Don't show any data. Texts are rendered by the item widget."""
        return QtCore.QVariant()

    def flags(self, index: QtCore.QModelIndex) -> Qt.ItemFlag:
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class QCommandLabel(QtW.QLabel):
    """The label widget to display a command in the palette."""

    DISABLED_COLOR = "gray"

    def __init__(self, cmd: Command | None = None):
        super().__init__()
        if cmd is not None:
            self.set_command(cmd)
        else:
            self._command = None
            self._command_text = ""

    def command(self) -> Command:
        """Command bound to this label."""
        return self._command

    def set_command(self, cmd: Command) -> None:
        """Set command to this widget."""
        command_text = cmd.fmt()
        self._command_text = command_text
        self._command = cmd
        self.setText(command_text)
        self.setToolTip(cmd.tooltip)

    def command_text(self) -> str:
        """The original command text."""
        return self._command_text

    def set_text_colors(self, input_text: str, color: str):
        """Set label text color based on the input text."""
        if input_text == "":
            return None
        text = self.command_text()
        words = input_text.split(" ")
        pattern = re.compile("|".join(words), re.IGNORECASE)

        output_texts: list[str] = []
        last_end = 0
        for match_obj in pattern.finditer(text):
            output_texts.append(text[last_end : match_obj.start()])
            word = match_obj.group()
            colored_word = bold_colored(word, color)
            output_texts.append(colored_word)
            last_end = match_obj.end()
        output_texts.append(text[last_end:])

        self.setText("".join(output_texts))
        return None

    def set_disabled(self) -> None:
        """Set the label to disabled."""
        text = self.command_text()
        self.setText(colored(text, self.DISABLED_COLOR))
        return None


class QCommandList(QtW.QListView):
    commandClicked = Signal(int)  # one of the items is clicked

    def __init__(
        self, viewer: ViewerModel, parent: QtW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setModel(QCommandMatchModel(self))
        self.setSelectionMode(QtW.QAbstractItemView.SelectionMode.NoSelection)
        self._selected_index = 0
        self._index_offset = (
            0  # NOTE: maybe useful for scrolling in the future
        )
        self._label_widgets: list[QCommandLabel] = []
        self._current_max_index = 0
        for i in range(self.model()._max_matches):
            lw = QCommandLabel()
            self._label_widgets.append(lw)
            self.setIndexWidget(self.model().index(i), lw)
        self.pressed.connect(self._on_clicked)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._match_color = "#468cc6"

    def match_color(self) -> str:
        return self._match_color

    def set_match_color(self, color: str):
        self._match_color = color

    def _on_clicked(self, index: QtCore.QModelIndex) -> None:
        if index.isValid():
            self.commandClicked.emit(index.row())
            return None

    def move_selection(self, dx: int) -> None:
        """Move selection by dx, dx can be negative or positive."""
        self._selected_index += dx
        self._selected_index = max(0, self._selected_index)
        self._selected_index = min(
            self._current_max_index - 1, self._selected_index
        )
        self.update_selection()
        return None

    def update_selection(self):
        index = self.model().index(self._selected_index - self._index_offset)
        self.selectionModel().setCurrentIndex(
            index, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        return None

    @property
    def all_commands(self) -> list[Command]:
        return self.model()._commands

    def add_command(self, command: Command) -> None:
        self.all_commands.append(command)
        return None

    def extend_command(self, commands: list[Command]) -> None:
        """Extend the list of commands."""
        self.all_commands.extend(commands)
        return None

    def clear_commands(self) -> None:
        """Clear all the command"""
        return self.all_commands.clear()

    def command_at(self, index: int) -> Command:
        i = index - self._index_offset
        return self.indexWidget(self.model().index(i)).command()

    def iter_command(self) -> Iterator[Command]:
        for i in range(self.model().rowCount()):
            if not self.isRowHidden(i):
                yield self.command_at(i)

    def execute(self, index: int | None = None) -> None:
        """Execute the currently selected command."""
        if index is None:
            index = self._selected_index
        cmd = self.command_at(index)
        if cmd is None:
            return None
        cmd()
        # move to the top
        self.all_commands.remove(cmd)
        self.all_commands.insert(0, cmd)
        return None

    def can_execute(self) -> bool:
        """Return true if the command can be executed."""
        index = self._selected_index
        cmd = self.command_at(index)
        return cmd.enabled()

    def update_for_text(self, input_text: str) -> None:
        """Update the list to match the input text."""
        self._selected_index = 0
        max_matches = self.model()._max_matches
        row = 0
        for cmd in self.iter_top_hits(input_text):
            self.setRowHidden(row, False)
            lw = self.indexWidget(self.model().index(row))
            lw.set_command(cmd)
            if cmd.enabled():
                lw.set_text_colors(input_text, color=self._match_color)
            else:
                lw.set_disabled()
            row += 1

            if row >= max_matches:
                self._current_max_index = max_matches
                break
        else:
            self._current_max_index = row
            for row in range(row, max_matches):
                self.setRowHidden(row, True)
        self.update_selection()
        self.update()
        return None

    def iter_top_hits(self, input_text: str) -> Iterator[Command]:
        """Iterate over the top hits for the input text"""
        for cmd in self.all_commands:
            if cmd.matches(input_text):
                yield cmd

    if TYPE_CHECKING:

        def model(self) -> QCommandMatchModel:
            ...

        def indexWidget(self, index: QtCore.QModelIndex) -> QCommandLabel:
            ...
