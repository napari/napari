"""TODO:"""

from PyQt5.QtWidgets import QVBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from napari.plugins import plugin_manager as napari_plugin_manager
from napari.utils.translations import trans


class QtReaderWriterList(QTableWidget):
    """ """

    def __init__(self, parent=None, type_: str = 'reader'):
        super().__init__(parent)
        self._type = type_

        rows = 0
        plugins = []
        hook_caller = getattr(
            napari_plugin_manager.hook, f'napari_get_{type_}', None
        )
        data = getattr(napari_plugin_manager, f"_extension2{type_}", None)
        if hook_caller is not None:
            for row, hook_implementation in enumerate(
                reversed(hook_caller._nonwrappers)
            ):
                if hook_implementation.plugin_name not in plugins:
                    plugins.append(hook_implementation.plugin_name)
                    rows += 1

        # Widget setup
        self.setColumnCount(2)
        self.setRowCount(rows)
        self.setHorizontalHeaderLabels(
            [trans._("Plugin name"), trans._("Extensions")]
        )
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setShowGrid(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStyleSheet(
            'border-bottom: 2px solid white;'
        )
        self.verticalHeader().setVisible(False)

        # Populate the table
        data = self._data2list(data)
        for row, plugin_name in enumerate(plugins):
            item = QTableWidgetItem(f"{row + 1}. {plugin_name}")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.setItem(row, 0, item)
            content = data.get(plugin_name)
            if content:
                item_exts = QTableWidgetItem(",".join(content))
                self.setItem(row, 1, item_exts)

        self.resizeColumnsToContents()
        self.cellChanged.connect(self._update)

    def _data2list(self, data: dict) -> dict:
        """Helper method to invert key,values of the extension data dict."""
        data = data or {}
        new_data = {}
        for key, value in data.items():
            if value not in new_data:
                new_data[value] = []

            new_data[value].append(key)
        return new_data

    def _update(self, row, column):
        """Update plugin manager."""
        plugin_name = self.item(row, 0).text().split(" ")[-1]
        extensions_item = self.item(row, column)
        if extensions_item is not None:
            extensions = extensions_item.text().split(',')
            extension_list = [
                it.strip() for it in extensions if it.strip()
            ]
            assign_to_extension = getattr(
                napari_plugin_manager,
                f"assign_{self._type}_to_extensions",
                None,
            )
            if assign_to_extension:
                try:
                    assign_to_extension(
                        plugin_name, extensions=extension_list
                    )
                    print(plugin_name, extension_list)
                except Exception as e:
                    print(e)


class QtReaderList(QWidget):
    """ """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._label = QLabel(
            trans._(
                "Add extensions separated by commas to directly read with plugin."
            )
        )
        self._table = QtReaderWriterList(parent=self, type_='reader')

        layout = QVBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._table)
        self.setLayout(layout)


class QtWriterList(QWidget):
    """ """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._label = QLabel(
            trans._(
                "Add extensions separated by commas to directly write with plugin."
            )
        )
        self._table = QtReaderWriterList(parent=self, type_='writer')

        layout = QVBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._table)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication([])
    w = QtReaderList()
    w.show()
    app.exec_()
