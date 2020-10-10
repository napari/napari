from qtpy.QtCore import Qt
from qtpy.QtWidgets import QAbstractItemView, QDialog, QLabel, QVBoxLayout

from ..widgets.qt_dict_table import QtDictTable


class QtPluginTable(QDialog):
    def __init__(self, parent, plugin_manager=None):
        super().__init__(parent)
        if not plugin_manager:
            from ...plugins import plugin_manager

        self.setMaximumHeight(800)
        self.setMaximumWidth(1280)
        layout = QVBoxLayout()
        # maybe someday add a search bar here?
        title = QLabel("Installed Plugins")
        title.setObjectName("h2")
        layout.addWidget(title)
        # get metadata for successfully registered plugins
        plugin_manager.discover()
        data = plugin_manager.list_plugin_metadata()
        data = list(filter(lambda x: x['plugin_name'] != 'builtins', data))
        # create a table for it
        self.table = QtDictTable(
            parent,
            data,
            headers=[
                'plugin_name',
                'package',
                'version',
                'url',
                'author',
                'license',
            ],
            min_section_width=60,
        )
        self.table.setObjectName("pluginTable")
        self.table.horizontalHeader().setObjectName("pluginTableHeader")
        self.table.verticalHeader().setObjectName("pluginTableHeader")
        self.table.setGridStyle(Qt.NoPen)
        # prevent editing of table
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.setAttribute(Qt.WA_DeleteOnClose)
