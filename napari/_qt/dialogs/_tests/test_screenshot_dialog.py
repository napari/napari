import pytest
from qtpy.QtWidgets import QLineEdit, QMessageBox

from napari._qt.dialogs.screenshot_dialog import ScreenshotDialog
from napari.utils.history import get_save_history


@pytest.mark.parametrize("filename", ["test", "test.png", "test.tif"])
def test_screenshot_save(qtbot, tmp_path, filename):
    """Check passing different extensions with the filename."""

    def save_function(path):
        assert filename in path
        if "." not in filename:
            assert ".png" in path

    dialog = ScreenshotDialog(
        save_function, directory=str(tmp_path), history=get_save_history()
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.windowTitle() == 'Save screenshot'
    line_edit = dialog.findChild(QLineEdit)
    line_edit.setText(filename)
    dialog.accept()


def test_screenshot_overwrite_save(qtbot, tmp_path, monkeypatch):
    """Check overwriting file validation."""
    filename = tmp_path / "test.png"
    filename.write_text("")

    def save_function(path):
        assert "test.png" in path

    def overwritte_message(*args):
        box, parent, title, text, buttons, default = args
        assert parent == dialog
        assert title == "Confirm overwrite"
        assert "test.png" in text
        assert "already exists. Do you want to replace it?" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default == QMessageBox.No

        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "warning", overwritte_message)

    dialog = ScreenshotDialog(
        save_function, directory=str(tmp_path), history=get_save_history()
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.windowTitle() == 'Save screenshot'
    line_edit = dialog.findChild(QLineEdit)
    line_edit.setText("test")
    dialog.accept()
