from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

from napari._qt.dialogs.qt_reader_dialog import get_reader_choice_for_file


class MockQtReaderDialog:
    """Dialog for user to select a reader plugin for a given file extension or folder"""

    def __init__(
        self,
        pth: str = '',
        parent: 'QWidget' = None,
        readers: Dict[str, str] = {},
        error_message: str = '',
        extension: str = '',
    ):
        self._current_file = pth
        self.readers = readers
        self.error_message = error_message
        self._extension = extension
        self._plugin_choice = None
        self._persist_choice = True
        self._cancelled = False

    def _set_plugin_choice(self, key):
        """Set plugin choice for use in tests"""
        self._plugin_choice = key

    def _set_persist_choice(self, persist: bool):
        """Set persistence checkbox choice for use in tests"""
        self._persist_choice = persist

    def _set_user_cancelled(self):
        """Mock that user chose to cancel on the dialog"""
        self._cancelled = True

    def get_user_choices(self) -> Optional[Tuple[str, bool]]:
        """Mock function for 'executing' dialog and getting user choices"""
        if self._cancelled:
            return

        if len(self.readers) == 1:
            return self._plugin_choice, False

        return self._plugin_choice, self._persist_choice


def test_get_reader_choice_single_reader():
    filename = './my_file.abc'
    readers = {'disp-name': 'plugin_name'}
    dialog = MockQtReaderDialog(filename, None, readers)
    choice = get_reader_choice_for_file(dialog, readers, False)

    assert choice[0] == 'disp-name'
    assert choice[1] is False


def test_get_reader_choice_cancel():
    filename = './my_file.abc'
    readers = {'disp-name': 'plugin_name', 'p1': 'p2'}
    dialog = MockQtReaderDialog(filename, None, readers)
    dialog._set_user_cancelled()

    choice = get_reader_choice_for_file(dialog, readers, False)
    assert choice is None


def test_get_reader_choice_many_persist():
    filename = './my_file.abc'
    readers = {'disp-name': 'plugin_name', 'p1': 'p2'}
    dialog = MockQtReaderDialog(filename, None, readers)
    dialog._set_plugin_choice('p1')

    choice = get_reader_choice_for_file(dialog, readers, False)
    assert choice[0] == 'p1'
    assert choice[1] is True


def test_get_reader_choice_no_persist():
    filename = './my_file.abc'
    readers = {'disp-name': 'plugin_name', 'p1': 'p2'}
    dialog = MockQtReaderDialog(filename, None, readers)
    dialog._set_plugin_choice('p1')
    dialog._set_persist_choice(False)

    choice = get_reader_choice_for_file(dialog, readers, False)
    assert choice[0] == 'p1'
    assert choice[1] is False
