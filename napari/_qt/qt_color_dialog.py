from .qt_modal import QtPopup
from qtpy.QtWidgets import QColorDialog
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor


class QColorPopup(QtPopup):
    currentColorChanged = Signal(QColor)
    colorSelected = Signal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_dialog = QColorDialog()

        # native dialog doesn't get added to the form_layout
        # so more would need to be done to use it
        self.color_dialog.setOption(QColorDialog.DontUseNativeDialog)
        self.form_layout.insertRow(0, self.color_dialog)
        self.setObjectName('QtModalPopup')
        self.color_dialog.currentColorChanged.connect(
            self.currentColorChanged.emit
        )
        self.color_dialog.colorSelected.connect(self._on_color_selected)
        self.color_dialog.rejected.connect(self._on_rejected)

    def _on_color_selected(self, color):
        self.colorSelected.emit(color)
        self.close()

    def _on_rejected(self):
        print("rejects")
        self.close()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            return self.color_dialog.accept()
        self.color_dialog.keyPressEvent(event)
