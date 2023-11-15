import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QFileDialog, QLineEdit, QMessageBox

from napari._qt.dialogs.screenshot_dialog import ScreenshotDialog


@pytest.mark.parametrize("filename", ["test", "test.png", "test.tif"])
def test_screenshot_save(qtbot, tmp_path, filename):
    """Check passing different extensions with the filename."""

    def save_function(path):
        # check incoming path has extension event when a filename without one
        # was provided
        assert filename in path
        assert "." in filename or ".png" in path

        # create a file with the given path to check for
        # non-native qt overwrite message
        with open(path, "w") as mock_img:
            mock_img.write("")

    qt_overwrite_shown = False

    def qt_overwrite_qmessagebox_warning():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMessageBox):  # pragma: no cover
                # test should not enter here!
                widget.accept()
                nonlocal qt_overwrite_shown
                qt_overwrite_shown = True
                break

    # setup dialog
    dialog = ScreenshotDialog(
        save_function, directory=str(tmp_path), history=[]
    )
    qtbot.addWidget(dialog)
    dialog.setOptions(QFileDialog.DontUseNativeDialog)
    dialog.show()

    # check dialog and set filename
    assert dialog.windowTitle() == 'Save screenshot'
    line_edit = dialog.findChild(QLineEdit)
    line_edit.setText(filename)

    # check that no warning message related with overwriting is shown
    QTimer.singleShot(100, qt_overwrite_qmessagebox_warning)
    dialog.accept()
    qtbot.wait(120)
    assert not qt_overwrite_shown, "Qt non-native overwrite message was shown!"

    # check the file was created
    save_filename = filename if '.' in filename else f'{filename}.png'
    qtbot.waitUntil((tmp_path / save_filename).exists)


def test_screenshot_overwrite_save(qtbot, tmp_path, monkeypatch):
    """Check overwriting file validation."""
    (tmp_path / "test.png").write_text("")

    def save_function(path):
        assert "test.png" in path
        (tmp_path / "test.png").write_text("overwritten")

    def overwrite_qmessagebox_warning(*args):
        box, parent, title, text, buttons, default = args
        assert parent == dialog
        assert title == "Confirm overwrite"
        assert "test.png" in text
        assert "already exists. Do you want to replace it?" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default == QMessageBox.No

        return QMessageBox.Yes

    # monkeypath custom overwrite QMessageBox usage
    monkeypatch.setattr(QMessageBox, "warning", overwrite_qmessagebox_warning)

    dialog = ScreenshotDialog(
        save_function, directory=str(tmp_path), history=[]
    )
    qtbot.addWidget(dialog)
    dialog.setOptions(QFileDialog.DontUseNativeDialog)
    dialog.show()

    # check dialog, set filename and trigger accept logic
    assert dialog.windowTitle() == 'Save screenshot'
    line_edit = dialog.findChild(QLineEdit)
    line_edit.setText("test")
    dialog.accept()

    # check the file was overwritten
    assert (tmp_path / "test.png").read_text() == "overwritten"
