import os

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)
from qtpy.QtWidgets import QDialog


class QtReaderDialog(QDialog):
    def __init__(
        self,
        pth=None,
        parent=None,
        readers=None,
        npe1_readers=None,
        error_message=None,
    ):
        super().__init__(parent)
        self.setObjectName('Choose reader')
        self.setWindowTitle('Choose reader')
        self._current_file = pth
        self._reader_buttons = []
        self.setup_ui(error_message, readers, npe1_readers)

    def setup_ui(self, error_message, readers, npe1_readers):
        layout = QVBoxLayout()
        self.setLayout(layout)

        label = QLabel()
        label.setText(
            f"{error_message if error_message else ''}Choose reader for file {self._current_file}:"
        )
        self.layout().addWidget(label)

        self.reader_btn_group = QButtonGroup(self)
        self.add_reader_buttons(readers)
        self.add_reader_buttons(npe1_readers)
        if self.reader_btn_group.buttons():
            self.reader_btn_group.buttons()[0].toggle()

        extension = os.path.splitext(self._current_file)[1]
        if extension:
            self.persist_checkbox = QCheckBox(
                f'Remember this choice for files with a {extension} extension'
            )
            self.persist_checkbox.toggle()
            self.layout().addWidget(self.persist_checkbox)

        btns = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.btn_box = QDialogButtonBox(btns)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        self.layout().addWidget(self.btn_box)

    def set_current_file(self, pth):
        self._current_file = pth

    def add_reader_buttons(self, readers):
        for reader in readers:
            button = QRadioButton(f"{reader}")
            self.reader_btn_group.addButton(button)
            self.layout().addWidget(button)

    def get_plugin_choice(self):
        return self.reader_btn_group.checkedButton().text()
