from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari.plugins.utils import get_all_readers

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
        self._set_up_table()
        self._edit_row = self._make_edit_row()
        self._populate_table()

        layout = QVBoxLayout()
        layout.addWidget(self._edit_row)
        layout.addWidget(self._table)
        self.setLayout(layout)

    def _set_up_table(self):
        """Add table columns and headers, define styling"""
        self._extension_col = 0
        self._reader_col = 1

        header_strs = [trans._('Extension'), trans._('Reader Plugin')]

        self._table.setColumnCount(2)
        self._table.setColumnWidth(self._extension_col, 175)
        self._table.setColumnWidth(self._reader_col, 250)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(120)
        # self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setStyleSheet(
            'border-bottom: 2px solid white;'
        )
        self._table.setHorizontalHeaderLabels(header_strs)

    def _populate_table(self):
        """Add row for each extension to reader mapping in settings"""

        extension2reader = get_settings().plugins.extension2reader
        if len(extension2reader) > 0:
            self._table.setRowCount(len(extension2reader))

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
                remove_btn = QPushButton('X', objectName=f'{extension}')
                remove_btn.setFixedWidth(30)
                remove_btn.setStyleSheet('margin: 4px;')
                remove_btn.setToolTip(
                    trans._('Remove this extension to reader association')
                )
                remove_btn.clicked.connect(self._remove_extension_assignment)

                plugin_widg.layout().addWidget(plugin_label)
                plugin_widg.layout().addWidget(remove_btn)
                self._table.setCellWidget(row, self._reader_col, plugin_widg)
        else:
            # Display that there are no extensions with reader associations
            self._table.setRowCount(1)
            item = QTableWidgetItem(trans._('No extensions found.'))
            item.setFlags(Qt.NoItemFlags)
            self._table.setItem(0, 0, item)

    def _make_edit_row(self):
        """Make row for user to add a new extension assignment"""
        edit_row_widget = QWidget()
        edit_row_widget.setLayout(QGridLayout())
        edit_row_widget.layout().setContentsMargins(0, 0, 0, 0)

        self._new_extension_edit = QLineEdit()
        self._new_extension_edit.setFixedWidth(175)
        self._new_extension_edit.setPlaceholderText(
            "Start typing file extension..."
        )

        add_reader_widg = QWidget()
        add_reader_widg.setLayout(QHBoxLayout())
        add_reader_widg.layout().setContentsMargins(0, 0, 0, 0)
        add_reader_widg.setFixedWidth(250)

        self._new_reader_dropdown = QComboBox()
        self._npe2_readers, self._npe1_readers = get_all_readers()
        for reader in self._npe2_readers.values():
            self._new_reader_dropdown.addItem(reader)

        add_btn = QPushButton('Add')
        add_btn.setFixedWidth(70)
        add_btn.setToolTip(trans._('Save reader preference for extension'))
        add_btn.clicked.connect(self._save_new_preference)

        add_reader_widg.layout().addWidget(self._new_reader_dropdown)
        add_reader_widg.layout().addWidget(add_btn)

        edit_row_widget.layout().addWidget(
            self._new_extension_edit,
            0,
            0,
        )
        edit_row_widget.layout().addWidget(add_reader_widg, 0, 1)

        return edit_row_widget

    def _save_new_preference(self, event):
        print(event)

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

    def value(self):
        """Return extension:reader mapping from settings.

        Returns
        -------
        Dict[str, str]
            mapping of extension to reader plugin display name
        """
        return get_settings().plugins.extension2reader
