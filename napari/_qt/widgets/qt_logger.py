import logging

from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari._qt.dialogs.qt_about import QtCopyToClipboardButton
from napari.utils._logging import LOG_STREAM, get_log_level_value
from napari.utils.translations import trans


class LogWidget(QWidget):
    """
    Widget for inspecting and filtering logging output.
    """

    def __init__(
        self,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(trans._('logger'))
        title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.layout.addWidget(title_label)

        # level selection
        self.level_layout = QHBoxLayout()
        self.level_layout.addWidget(QLabel('Current log level:'))
        self.loglevel = QComboBox()
        self.loglevel.addItems(
            ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        )
        self.level_layout.addWidget(self.loglevel)
        self.level_layout.addStretch()
        self.layout.addLayout(self.level_layout)

        # filtering
        self.filter_layout = QHBoxLayout()
        self.filter_layout.addWidget(QLabel('Filter:'))
        self.level_filter = QComboBox()
        self.level_filter.addItems(
            ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        )
        self.filter_layout.addWidget(self.level_filter)

        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText('text filter')
        self.filter_layout.addWidget(self.text_filter)
        self.layout.addLayout(self.filter_layout)

        # log text box
        self.log_text_box = QTextEdit()
        self.log_text_box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.log_text_box.setLineWrapMode(QTextEdit.NoWrap)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.log_text_box.setFont(font)

        # Add text copy button
        self.infoCopyButton = QtCopyToClipboardButton(self.log_text_box)
        self.info_layout = QHBoxLayout()
        self.info_layout.addWidget(self.log_text_box, 1)
        self.info_layout.addWidget(
            self.infoCopyButton, 0, Qt.AlignmentFlag.AlignTop
        )
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addLayout(self.info_layout)

        self.setLayout(self.layout)

        self.loglevel.currentTextChanged.connect(self._on_loglevel_change)
        self.level_filter.currentTextChanged.connect(self._on_change)
        self.text_filter.textChanged.connect(self._on_change)

        # TODO: super laggy when open :/
        LOG_STREAM.changed.connect(self._on_new_message)
        self._prev_pos = None
        self.log_text_box.verticalScrollBar().rangeChanged.connect(
            self._jump_to_pos
        )

        self._on_loglevel_change()
        self._on_change()
        self._jump_to_pos()

    def _on_loglevel_change(self, event=None):
        level = get_log_level_value(event)
        logging.getLogger().setLevel(level)

    def _on_new_message(self, event=None):
        self._prev_pos = self._scroll_pos()

        log = LOG_STREAM.get_filtered_logs_html(
            self.level_filter.currentText(),
            self.text_filter.text(),
            last_only=True,
        )[0]
        self.log_text_box.append(log)

    def _on_change(self, event=None):
        self._prev_pos = self._scroll_pos()

        logs = LOG_STREAM.get_filtered_logs_html(
            self.level_filter.currentText(), self.text_filter.text()
        )
        self.log_text_box.clear()
        for log in logs:
            # by looping here instead of joinging the lines,
            # we ensure each line is separate (allows better selection)
            self.log_text_box.append(log)

    def _jump_to_pos(self, event=None):
        # maintains position when updating the contents of the text
        # for some reason using scrollbar.setValue() doesn't keep up,
        # cursor's better
        scrollbar = self.log_text_box.verticalScrollBar()
        if self._prev_pos is None:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(self._prev_pos)

        # self.log_text_box.moveCursor(self.log_text_box.textCursor().End)
        # self.log_text_box.moveCursor(self.log_text_box.textCursor().StartOfLine)
        # self.log_text_box.ensureCursorVisible()

    def _scroll_pos(self):
        scrollbar = self.log_text_box.verticalScrollBar()
        curr = scrollbar.value()
        if curr == scrollbar.maximum():
            return None
        return curr
