from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...settings import get_settings
from ...utils.translations import trans


class Extension2ReaderTable(QWidget):
    """Table showing extension to reader mappings with removal button.

    Widget presented in preferences-plugin dialog."""

    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._table = QTableWidget()
        self._table.setShowGrid(False)
        self._populate_table()

        layout = QVBoxLayout()
        layout.addWidget(self._table)
        self.setLayout(layout)

    def _populate_table(self):
        """Add row for each extension to reader mapping in settings"""
        self._extension_col = 0
        self._reader_col = 1

        header_strs = [trans._('Extension'), trans._('Reader Plugin')]

        self._table.setColumnCount(2)
        self._table.setColumnWidth(self._extension_col, 100)
        self._table.setColumnWidth(self._reader_col, 150)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(120)

        extension2reader = get_settings().plugins.extension2reader
        if len(extension2reader) > 0:
            self._table.setRowCount(len(extension2reader))
            self._table.horizontalHeader().setStretchLastSection(True)
            self._table.horizontalHeader().setStyleSheet(
                'border-bottom: 2px solid white;'
            )
            self._table.setHorizontalHeaderLabels(header_strs)

            for row, (extension, plugin_name) in enumerate(
                extension2reader.items()
            ):
                item = QTableWidgetItem(extension)
                item.setFlags(Qt.NoItemFlags)
                self._table.setItem(row, self._extension_col, item)

                plugin_widg = QWidget()
                # need object name to easily find row
                plugin_widg.setObjectName(f'{extension}')
                plugin_widg.setLayout(QHBoxLayout())
                plugin_widg.layout().setContentsMargins(0, 0, 0, 0)

                plugin_label = QLabel(plugin_name)
                # need object name to easily work out which button was clicked
                remove_btn = QPushButton('x', objectName=f'{extension}')
                remove_btn.setFixedWidth(30)
                remove_btn.setStyleSheet('margin: 4px;')
                remove_btn.setToolTip(
                    'Remove this extension to reader association'
                )
                remove_btn.clicked.connect(self._remove_extension_assignment)

                plugin_widg.layout().addWidget(plugin_label)
                plugin_widg.layout().addWidget(remove_btn)
                self._table.setCellWidget(row, self._reader_col, plugin_widg)
        else:
            # Display that there are no extensions with reader associations
            self._table.setRowCount(1)
            self._table.setHorizontalHeaderLabels(header_strs)

            self._table.setColumnHidden(self._reader_col, True)
            self._table.setColumnWidth(self._extension_col, 200)
            item = QTableWidgetItem(trans._('No extensions found.'))
            item.setFlags(Qt.NoItemFlags)
            self._table.setItem(0, 0, item)

    def _remove_extension_assignment(self, event):
        """Delete extension to reader mapping setting and remove table row"""
        extension_to_remove = self.sender().objectName()
        current_settings = get_settings().plugins.extension2reader
        # need explicit assignment to new object here for persistence
        get_settings().plugins.extension2reader = {
            k: v
            for k, v in current_settings.items()
            if k != extension_to_remove
        }

        for i in range(self._table.rowCount()):
            row_widg_name = self._table.cellWidget(
                i, self._reader_col
            ).objectName()
            if row_widg_name == extension_to_remove:
                self._table.removeRow(i)
                return
