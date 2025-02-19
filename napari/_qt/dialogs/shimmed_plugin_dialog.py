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
        cancel_btn = QPushButton(trans._('Cancel'))
        close_btn = QPushButton(trans._('Close'))
        icon_label = QWidget()
        icon_label.setObjectName('warning_icon_element')
        plugin_info_text = """
The following plugins are using the old plugin engine and have been
automatically converted to the new plugin engine:
        """
        plugin_text = '\n'.join(plugins)
        info_text = """
Some functionality e.g. code designed to run on import may not work as expected.

If you notice broken functionality in any of these plugins, turn off
the 'Use npe2 adaptor' setting in the plugin preferences.

This setting will cease to exist in napari 0.7.0 and these plugins
will only be usable if they can be automatically converted.

Please contact the plugin author to ask them about updating their plugin to use npe2,
the new plugin engine.
        """
        self.only_new_checkbox = QCheckBox(
            trans._('Only warn me about newly installed plugins.')
        )

        cancel_btn.clicked.connect(self.reject)
        close_btn.clicked.connect(self.accept)

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
        layout4.addWidget(cancel_btn)
        layout4.addWidget(close_btn)
        layout.addLayout(layout2)
        layout.addLayout(layout3)
        layout.addLayout(layout4)
        self.setLayout(layout)

        # for test purposes because of problem with shortcut testing:
        # https://github.com/pytest-dev/pytest-qt/issues/254
        self.close_btn = close_btn
        self.cancel_btn = cancel_btn

    def accept(self) -> None:
        if self.only_new_checkbox.isChecked():
            get_settings().plugins.warn_on_shimmed_plugin = (
                PluginShimWarningLevel.NEW
            )
        super().accept()
