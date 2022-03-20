import os
from typing import Dict, Optional, Tuple

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ...utils.translations import trans


class QtReaderDialog(QDialog):
    """Dialog for user to select a reader plugin for a given file extension or folder"""

    def __init__(
        self,
        pth: str = '',
        parent: QWidget = None,
        extension: str = '',
        readers: Dict[str, str] = {},
        error_message: str = '',
    ):
        super().__init__(parent)
        self.setObjectName('Choose reader')
        self.setWindowTitle(trans._('Choose reader'))
        self._current_file = pth
        self._extension = extension
        self._reader_buttons = []
        self.setup_ui(error_message, readers)

    def setup_ui(self, error_message, readers):
        """Build UI using given error_messsage and readers dict"""

        # add instruction label
        layout = QVBoxLayout()
        label = QLabel(
            f"{error_message}Choose reader for {self._current_file}:"
        )
        layout.addWidget(label)

        # add radio button for each reader plugin
        self.reader_btn_group = QButtonGroup(self)
        self.add_reader_buttons(layout, readers)
        if self.reader_btn_group.buttons():
            self.reader_btn_group.buttons()[0].toggle()

        # OK & cancel buttons for the dialog
        btns = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.btn_box = QDialogButtonBox(btns)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)

        # checkbox to remember the choice (doesn't pop up for folders)
        extension = os.path.splitext(self._current_file)[1]
        if extension:
            self.persist_checkbox = QCheckBox(
                f'Remember this choice for files with a {extension} extension'
            )
            self.persist_checkbox.toggle()
            layout.addWidget(self.persist_checkbox)

        layout.addWidget(self.btn_box)
        self.setLayout(layout)

    def add_reader_buttons(self, layout, readers):
        """Add radio button to layout for each reader in readers"""
        for display_name in sorted(readers.values()):
            button = QRadioButton(f"{display_name}")
            self.reader_btn_group.addButton(button)
            layout.addWidget(button)

    def _get_plugin_choice(self):
        """Get user's plugin choice based on the checked button"""
        checked_btn = self.reader_btn_group.checkedButton()
        if checked_btn:
            return checked_btn.text()

    def _get_persist_choice(self):
        """Get persistence checkbox choice"""
        return (
            hasattr(self, 'persist_checkbox')
            and self.persist_checkbox.isChecked()
        )

    def get_user_choices(self) -> Optional[Tuple[str, bool]]:
        """Execute dialog and get user choices"""
        dialog_result = self.exec_()
        # user pressed cancel
        if not dialog_result:
            return None

        # grab the selected radio button text
        display_name = self._get_plugin_choice()
        # grab the persistence checkbox choice
        persist_choice = self._get_persist_choice()
        return display_name, persist_choice


def get_reader_helper(_pth, viewer):
    _, extension = os.path.splitext(_pth)

    def select_reader_helper(_path, readers, exception):
        error_message = ''
        if exception:
            error_message += f"{str(exception)}\n"
        readerDialog = QtReaderDialog(
            parent=viewer,
            pth=_path,
            extension=extension,
            error_message=error_message,
            readers=readers,
        )
        display_name, persist_choice = get_preferred_reader(
            readerDialog, readers
        )
        return display_name, persist_choice

    return select_reader_helper


def get_preferred_reader(readerDialog, readers):
    """Get preferred reader from user through dialog

    Parameters
    ----------
    readerDialog : QtReaderDialog
        dialog for user to select their preferences
    readers : Dict[str, str]
        dictionary of plugin_name:display_name of available readers
    """
    display_name = ''
    persist_choice = False

    res = readerDialog.get_user_choices()
    if res:
        display_name, persist_choice = res[0], res[1]
    return display_name, persist_choice
