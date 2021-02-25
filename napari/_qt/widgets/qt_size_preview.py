from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIntValidator, QFont
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QHBoxLayout,
    QDialog,
    QWidget,
    QSlider,
    QLineEdit,
    QGridLayout,
    QVBoxLayout,
    QPlainTextEdit,
    QTextEdit,
)

from ...utils.translations import translator


trans = translator.load()


class QtFontSizePreview(QFrame):
    """
    """

    def __init__(self, parent):
        super().__init__(parent)

        self._text = ""

        # Widget
        self._preview = QPlainTextEdit(self)

        # Widget setup
        self._preview.setReadOnly(True)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self._preview)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def setText(self, text):
        self._preview.setPlainText(text)


class SizeSliderPreviewWidget(QDialog):
    """
    """
    valueChanged = Signal(int)

    def __init__(
            self,
            parent: QWidget = None,
            description :str = "",
            preview_text :str = "",
            value: int = None,
            min_value:int = 1,
            max_value:int = 50,
            unit:str = "px",
        ):
        super().__init__(parent)

        self._value = value if value else self.fontMetrics().height()
        self._min_value = min_value
        self._max_value = max_value

        # Widget
        self._lineedit = QLineEdit()
        self._description = QLabel(self)
        self._unit = QLabel(self)
        self._slider = QSlider(Qt.Horizontal, self)
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        self._preview = QtFontSizePreview(self)
        self._preview_label = QLabel(self)
        self._validator = QIntValidator(min_value, max_value, self)

        # Widgets setup
        self._description.setText(description)
        self._description.setWordWrap(True)
        self._unit.setText(unit)
        self._lineedit.setValidator(self._validator)
        self._lineedit.setAlignment(Qt.AlignRight)
        self._slider_min_label.setText(str(min_value))
        self._slider_max_label.setText(str(max_value))
        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)
        self._preview.setText(preview_text)
        self._preview_label.setText(trans._("preview"))
        self._preview_label.setAlignment(Qt.AlignHCenter)

        # Signals
        self._slider.valueChanged.connect(self._update_value)
        self._lineedit.textChanged.connect(self._update_value)

        # Layout
        # self._layout = QGridLayout()
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self._description)

        left_bottom_layout = QHBoxLayout()
        left_bottom_layout.addWidget(self._lineedit)
        left_bottom_layout.addWidget(self._unit)
        left_bottom_layout.addStretch()
        left_bottom_layout.addWidget(self._slider_min_label)
        left_bottom_layout.addWidget(self._slider)
        left_bottom_layout.addWidget(self._slider_max_label)

        left_layout.addLayout(left_bottom_layout)

        layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self._preview)
        right_layout.addWidget(self._preview_label)

        layout.addLayout(right_layout)

        self.setLayout(layout)
        self._update_line_width()

        self.refresh()

    def _update_line_width(self):
        size = self._lineedit.fontMetrics().width("m" * (1 + len(str(self._max_value))))
        self._lineedit.setMaximumWidth(size)
        self._lineedit.setMinimumWidth(size)

    def _update_value(self, value):
        if value == "":
            value = int(self._value)

        self._value = int(value)
        self.refresh()

    def refresh(self):
        self.blockSignals(True)
        self._lineedit.setText(str(self._value))
        self._slider.setValue(self._value)
        font = QFont()
        font.setPixelSize(self._value)
        self._preview.setFont(font)
        self.blockSignals(False)
        self.valueChanged.emit(self._value)

    def getValue(self):
        return self._value

    def setValue(self, value):
        self._update_value(value)
        self._refresh()

    def getDescription(self):
        return self._desctiption_label.text()

    def setDescription(self, text):
        self._desctiption_label.setText(text)

    def getPreviewText(self):
        return self._preview.text()

    def setPreviewText(self, text):
        self._preview.setText(text)

    def getUnit(self):
        return self._unit_label.text()

    def setUnit(self, text):
        self._unit_label.setText(text)

    def setMinimun(self, value):
        """
        """
        value = int(value)
        if value < self._max_value:
            self._min_value = value
            self._slider_min_label.setText(str(value))
            self._value = self._min_value if self._value < self._min_value else self._value
            self.refresh()
        else:
            raise ValueError(f"Minimum value must be smaller than {self._max_value}")

    def getMinimun(self, value):
        """
        """
        return self._min_value

    def setMaximum(self, value):
        """
        """
        value = int(value)
        if value > self._max_value:
            self._max_value = value
            self._slider_max_label.setText(str(value))
            self._value = self._max_value if self._value > self._max_value else self._value
            self._update_line_width()
            self.refresh()
        else:
            raise ValueError(f"Maximum value must be larger than {self._min_value}")

    def getMaximum(self, value):
        """
        """
        return self._max_value
