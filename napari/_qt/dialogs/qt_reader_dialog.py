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

from napari.plugins.utils import get_potential_readers
from napari.settings import get_settings

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
            f"{error_message}\nChoose reader for {self._current_file}:"
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
        for display_name in sorted(readers.keys()):
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


def handle_gui_reading(_paths, viewer, stack, plugin, error):
    _path = _paths[0]
    _, extension = os.path.splitext(_path)

    readers = get_potential_readers(_path)
    # remove plugin we already tried e.g. prefered plugin
    if plugin in readers:
        del readers[plugin]
    # if there's no other readers left, raise the exception
    if not readers:
        raise error

    error_message = str(error)
    # showing this message in the GUI would be redundant so we clear it
    if 'Multiple plugins found' in error_message:
        error_message = ''

    readerDialog = QtReaderDialog(
        parent=viewer,
        # TODO: we probably want the reader dialog to take all paths and just display them nicely somehow
        pth=_path,
        extension=extension,
        error_message=error_message,
        readers=readers,
    )
    display_name, persist = get_reader_from_dialog(readerDialog)
    if display_name:
        # TODO: disambiguate with reader title
        plugin_name = [
            p_name
            for d_name, p_name in readers.items()
            if d_name == display_name
        ][0]
        # may throw error, but we let it this time
        viewer.viewer._add_layers_with_plugins(
            _paths, stack=stack, plugin=plugin_name
        )

        if persist:
            get_settings().plugins.extension2reader = {
                **get_settings().plugins.extension2reader,
                extension: display_name,
            }


def get_reader_from_dialog(readerDialog):
    """Get preferred reader from user through dialog

    Parameters
    ----------
    readerDialog : QtReaderDialog
        dialog for user to select their preferences
    """
    display_name = ''
    persist_choice = False

    res = readerDialog.get_user_choices()
    if res:
        display_name, persist_choice = res[0], res[1]
    return display_name, persist_choice
