import os
from pathlib import Path
from typing import Any, Callable

from qtpy.QtWidgets import QFileDialog, QMessageBox


class ScreenshotDialog(QFileDialog):
    """
    Dialog to chose save location of screenshot.

    Parameters
    ----------
    save_function : Callable[[str], Any],
        Function to be called on success of selecting save location
     parent : QWidget, optional
        Optional parent widget for this widget..
    directory : str, optional
        Starting directory to be set to File Dialog

    """

    def __init__(
        self,
        save_function: Callable[[str], Any],
        parent=None,
        directory=str(Path.home()),
    ):
        super().__init__(parent, "Save screenshot")
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setFileMode(QFileDialog.AnyFile)
        self.setNameFilter("Image files (*.png *.bmp *.gif *.tif *.tiff)")
        self.setDirectory(directory)

        self.save_function = save_function

    def accept(self):
        save_path = self.selectedFiles()[0]
        if os.path.splitext(save_path)[1] == "":
            save_path = save_path + ".png"
            if os.path.exists(save_path):
                res = QMessageBox().warning(
                    self,
                    "Confirm overwrite",
                    f"{save_path} already exists. Do you want to replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res != QMessageBox.Yes:
                    # standard accept return 1, reject 0. This inform that dialog should be reopened
                    super().accept()
                    self.exec_()
        self.save_function(save_path)
        return super().accept()
