from .qt_modal import QtPopup
from qtpy.QtWidgets import QColorDialog
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor


class MyColorDialog(QColorDialog):
    def keyPressEvent(self, event):
        event.ignore()


class QColorPopup(QtPopup):
    currentColorChanged = Signal(QColor)
    colorSelected = Signal(QColor)

    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        self.color_dialog = MyColorDialog(self)
        # native dialog doesn't get added to the form_layout
        # so more would need to be done to use it
        self.color_dialog.setOptions(QColorDialog.DontUseNativeDialog)
        self.form_layout.insertRow(0, self.color_dialog)
        self.setObjectName('QtColorPopup')
        self.color_dialog.currentColorChanged.connect(
            self.currentColorChanged.emit
        )
        self.color_dialog.colorSelected.connect(self._on_color_selected)
        self.color_dialog.rejected.connect(self._on_rejected)
        self.color_dialog.setCurrentColor(QColor(initial))

    def _on_color_selected(self, color):
        self.colorSelected.emit(color)
        self.close()

    def _on_rejected(self):
        self.close()

    def keyPressEvent(self, event):
        print('press')
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            print('enter')
            return self.color_dialog.accept()
        self.color_dialog.keyPressEvent(event)
