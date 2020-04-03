from qtpy.QtWidgets import QTableWidget, QTableWidgetItem
from qtpy.QtCore import Slot, QSize
from typing import List, Dict, Optional
import re
import webbrowser

email_pattern = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
url_pattern = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}"
    r"\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


class QtDictTable(QTableWidget):
    def __init__(
        self,
        parent=None,
        source=None,
        *,
        headers=None,
        min_section_width=None,
        max_section_width=480,
    ):
        super().__init__(parent=parent)
        if min_section_width:
            self.horizontalHeader().setMinimumSectionSize(min_section_width)
        self.horizontalHeader().setMaximumSectionSize(max_section_width)
        self.horizontalHeader().setStretchLastSection(True)
        if source:
            self.set_data(source, headers)
        self.cellDoubleClicked.connect(self._on_double_click)
        self.setMouseTracking(True)

    def set_data(
        self, data: List[Dict[str, str]], headers: Optional[List[str]] = None
    ):
        nrows = len(data)
        _headers = sorted(set().union(*data))
        if headers:
            for h in headers:
                if h not in _headers:
                    raise ValueError(
                        f"Argument 'headers' got item {h}, which was "
                        "not found in any of the items in 'data'"
                    )
            _headers = headers
        self.setRowCount(nrows)
        self.setColumnCount(len(_headers))
        for row, elem in enumerate(data):
            for key, value in elem.items():
                if key not in _headers:
                    continue
                col = _headers.index(key)
                self.setItem(row, col, QTableWidgetItem(value))

        self.setHorizontalHeaderLabels(_headers)
        self.resize_to_fit()

    @Slot(int, int)
    def _on_double_click(self, row, col):
        item = self.item(row, col)
        text = item.text().strip()
        if email_pattern.match(text):
            webbrowser.open(f'mailto:{text}', new=1)
            return
        if url_pattern.match(text):
            webbrowser.open(text, new=1)

    def resize_to_fit(self):
        self.resizeColumnsToContents()
        self.resize(self.sizeHint())

    def sizeHint(self):
        width = sum(map(self.columnWidth, range(self.columnCount()))) + 25
        height = self.rowHeight(0) * (self.rowCount() + 1)
        return QSize(width, height)
