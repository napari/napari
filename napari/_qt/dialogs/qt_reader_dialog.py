import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget
    from typing import Dict, Optional, Tuple

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)


class QtReaderDialog(QDialog):
    """Dialog for user to select a reader plugin for a given file extension or folder"""

    def __init__(
        self,
        pth: str = '',
        parent: "QWidget" = None,
        readers: "Dict[str, str]" = {},
        error_message: str = '',
    ):
        super().__init__(parent)
        self.setObjectName('Choose reader')
        self.setWindowTitle('Choose reader')
        self._current_file = pth
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
        for display_name in sorted(readers):
            button = QRadioButton(f"{display_name}")
            self.reader_btn_group.addButton(button)
            layout.addWidget(button)

    def get_plugin_choice(self):
        """Get user's plugin choice based on the checked button"""
        checked_btn = self.reader_btn_group.checkedButton()
        if checked_btn:
            return checked_btn.text()

    def get_user_choices(self) -> "Optional[Tuple[str, bool]]":
        dialog_result = self.exec_()
        # user pressed cancel
        if not dialog_result:
            return

        # grab the selected radio button text
        display_name = self.get_plugin_choice()
        # is setting being persisted?
        persist_choice = (
            hasattr(self, 'persist_checkbox')
            and self.persist_checkbox.isChecked()
        )
        return display_name, persist_choice
