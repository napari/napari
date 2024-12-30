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
    def __init__(self, parent, plugins: set[str]) -> None:
        super().__init__(parent)
        cancel_btn = QPushButton(trans._('Cancel'))
        close_btn = QPushButton(trans._('Close'))
        icon_label = QWidget()
        icon_label.setObjectName('warning_icon_element')
        plugin_info_text = 'The following plugins are npe1 plugins that have been shimmed to work with npe2: '
        plugin_text = '\n'.join(plugins)
        info_text = """
Some functionality e.g. code designed to run on import may not work as expected.

To use this plugin without shimming, turn off the 'Use npe2 adaptor' setting in the plugin preferences.
This setting will cease to exist in napari 0.0.6 npe1 plugins will only be usable in shimmed form.

Please contact the plugin author to ask them about updating their plugin to npe2.
"""
        self.do_not_ask = QCheckBox(
            trans._('Never warn me about shimmed plugins.')
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
        layout4.addWidget(self.do_not_ask)
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

    def accept(self):
        if self.do_not_ask.isChecked():
            get_settings().plugins.warn_on_shimmed_plugin = (
                PluginShimWarningLevel.NEVER
            )
        super().accept()
