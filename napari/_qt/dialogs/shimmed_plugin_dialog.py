from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.settings import get_settings
from napari.settings._constants import PluginShimWarningLevel
from napari.utils.translations import trans


class ShimmedPluginDialog(QDialog):
    def __init__(self, parent: QWidget, plugins: set[str]) -> None:
        super().__init__(parent)
        okay_btn = QPushButton(trans._('Okay'))
        icon_label = QWidget()
        icon_label.setObjectName('warning_icon_element')
        plugin_info_text = """
The following plugins use napari's old plugin engine. These plugins have been automatically converted to the new plugin engine, npe2:
        """
        plugin_text = '\n'.join(plugins)
        info_text = """
While the conversion should work for these plugins, some plugin functionality, such as code designed to run on import, may not work as expected.

If you notice an error in any of these plugins, drop back to using the old plugin engine by turning off the 'Use npe2 adaptor' setting in the plugin preferences.

The 'Use npe2 adaptor' setting will be removed in napari 0.7.0, and plugins will only be usable if they can be automatically converted or updated to use npe2.

For plugin upgrades, contact the plugin's author and request npe2 updates.
        """
        self.only_new_checkbox = QCheckBox(
            trans._('Only warn me about newly installed plugins.')
        )
        new_only = (
            get_settings().plugins.warn_on_shimmed_plugin
            == PluginShimWarningLevel.NEW
        )
        self.only_new_checkbox.setChecked(new_only)

        okay_btn.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout2.addWidget(icon_label)
        layout2a = QVBoxLayout()
        layout2a.addWidget(QLabel(plugin_info_text))
        layout2a.addWidget(QLabel(plugin_text))
        layout2.addLayout(layout2a)
        layout3 = QVBoxLayout()
        layout3.addWidget(QLabel(info_text))
        layout4 = QHBoxLayout()
        layout4.addWidget(self.only_new_checkbox)
        layout4.addStretch(1)
        layout4.addWidget(okay_btn)
        layout.addLayout(layout2)
        layout.addLayout(layout3)
        layout.addLayout(layout4)
        self.setLayout(layout)

        # for test purposes because of problem with shortcut testing:
        # https://github.com/pytest-dev/pytest-qt/issues/254
        self.okay_btn = okay_btn

    def accept(self) -> None:
        if self.only_new_checkbox.isChecked():
            get_settings().plugins.warn_on_shimmed_plugin = (
                PluginShimWarningLevel.NEW
            )
        else:
            get_settings().plugins.warn_on_shimmed_plugin = (
                PluginShimWarningLevel.ALWAYS
            )
        super().accept()
