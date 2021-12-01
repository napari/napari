from qtpy import PYQT5
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QMessageBox,
    QVBoxLayout,
    QCheckBox,
    QSpacerItem,
)


class QtMessageCheckBox(QMessageBox):
    """
    A QMessageBox derived widget that includes a QCheckBox aligned to the right
    under the message and on top of the buttons.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowModality(Qt.NonModal)
        self._checkbox = QCheckBox(self)

        # Set layout to include checkbox
        size = 9
        check_layout = QVBoxLayout()
        check_layout.addItem(QSpacerItem(size, size))
        check_layout.addWidget(self._checkbox, 0, Qt.AlignRight)
        check_layout.addItem(QSpacerItem(size, size))

        # Access the Layout of the QMessageBox to add the QCheckBox
        layout = self.layout()
        if PYQT5:
            layout.addLayout(check_layout, 1, 2)
        else:
            layout.addLayout(check_layout, 1, 1)

    def is_checked(self):
        return self._checkbox.isChecked()

    def set_checked(self, value):
        return self._checkbox.setChecked(value)

    def set_check_visible(self, value):
        self._checkbox.setVisible(value)

    def is_check_visible(self):
        self._checkbox.isVisible()

    def checkbox_text(self):
        self._checkbox.text()

    def set_checkbox_text(self, text):
        self._checkbox.setText(text)
