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
from napari.utils.translations import trans


class ShimmedPluginDialog(QDialog):
    def __init__(self, parent: QWidget, plugins: set[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle('Installed Plugin Warning')
        okay_btn = QPushButton(trans._('Okay'))
        icon_label = QWidget()
        icon_label.setObjectName('warning_icon_element')
        self.plugin_info_text = """
The following plugins use napari's deprecated plugin engine. These plugins have been automatically converted to the new plugin engine, npe2:
        """
        self.plugins = plugins
        self.plugin_text = '\n'.join(sorted(self.plugins))
        self.info_text = """
Conversion should work for most plugins. Some plugin functionality, such as code designed to run on import, may not work as expected.

If you see an error with these plugins, disable the 'Use npe2 adaptor' setting in plugin preferences. This will drop back napari to use the deprecated plugin engine.

The 'Use npe2 adaptor' setting will be removed in napari 0.7.0. Only npe2-compatible plugins will be supported.

For plugin upgrades, contact the plugin's author and request npe2 updates.
        """
        self.only_new_checkbox = QCheckBox(
            trans._('Only warn me about newly installed plugins.')
        )
        new_only = get_settings().plugins.only_new_shimmed_plugins_warning
        self.only_new_checkbox.setChecked(new_only)

        okay_btn.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout2.addWidget(icon_label)
        layout2a = QVBoxLayout()
        layout2a.addWidget(QLabel(self.plugin_info_text))
        layout2a.addWidget(QLabel(self.plugin_text))
        layout2.addLayout(layout2a)
        layout3 = QVBoxLayout()
        layout3.addWidget(QLabel(self.info_text))
        layout4 = QHBoxLayout()
        layout4.addWidget(self.only_new_checkbox)
        layout4.addStretch(1)
        layout4.addWidget(okay_btn)
        layout.addLayout(layout2)
        layout.addLayout(layout3)
        layout.addLayout(layout4)
        self.setLayout(layout)

        # To support shortcut testing and its limitations, the okay button is set:
        # https://github.com/pytest-dev/pytest-qt/issues/254
        self.okay_btn = okay_btn

    def accept(self) -> None:
        if self.only_new_checkbox.isChecked():
            get_settings().plugins.only_new_shimmed_plugins_warning = True
            get_settings().plugins.already_warned_shimmed_plugins.update(
                self.plugins
            )
        else:
            get_settings().plugins.only_new_shimmed_plugins_warning = False
            get_settings().plugins.already_warned_shimmed_plugins.clear()
        super().accept()
