
from PyQt5.QtWidgets import QButtonGroup, QCheckBox, QLabel, QRadioButton, QVBoxLayout, QDialogButtonBox
from qtpy.QtWidgets import QDialog


class QtReaderDialog(QDialog):
    def __init__(self, pth=None, parent=None, error_message=None):
        super().__init__(parent)
        self.setObjectName('Choose reader')
        self.setWindowTitle('Choose reader')
        self._current_file = pth
        self._reader_buttons = []
        self.setup_ui(error_message)

    def setup_ui(self, error_message):
        layout = QVBoxLayout()
        self.setLayout(layout)

        label = QLabel()
        label.setText(f"{error_message if error_message else ''}Choose reader for file {self._current_file}:")
        self.layout().addWidget(label)

        # TODO: move this back out to qt_viewer?
        from ...plugins import _npe2, plugin_manager
        readers = _npe2.get_readers(self._current_file)

        npe1_readers = []
        for spec, hook_caller in plugin_manager.hooks.items():
            if spec == 'napari_get_reader':
                potential_readers = hook_caller.get_hookimpls()
                for get_reader in potential_readers:
                    reader = hook_caller._call_plugin(get_reader.plugin_name, path=self._current_file)
                    if callable(reader):
                        npe1_readers.append(get_reader.plugin_name)

        self.reader_btn_group = QButtonGroup(self)
        self.add_reader_buttons(readers)
        self.add_reader_buttons(npe1_readers)
        if self.reader_btn_group.buttons():
            self.reader_btn_group.buttons()[0].toggle()

        # TODO: will fail with no extension
        self.persist_checkbox = QCheckBox(f'Remember this choice for files with a .{self._current_file.split(".")[1]} extension')
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


