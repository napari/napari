import sys
from collections import namedtuple

from qtpy.QtCore import QSettings, Signal
from qtpy.QtWidgets import QDialog, QGridLayout, QLabel

from .auto_widget import val_to_widget

SETTINGS = QSettings('napari', 'napari')

SetTup = namedtuple("Setting", ["key", "default", "description"])

RESTORE_GEOMETRY = SetTup(
    "prefs/restore_geometry",
    True,
    "Preserve window size/position across sessions",
)

this = sys.modules[__name__]
OPTIONS = {k: v for k, v in this.__dict__.items() if isinstance(v, SetTup)}
for settup in OPTIONS.values():
    if not SETTINGS.contains(settup.key):
        SETTINGS.setValue(settup.key, settup.default)


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()


class PreferencesWindow(QDialog):
    def __init__(self,):
        super().__init__()
        self.setLayout(QGridLayout())
        self.setObjectName("preferencesWindow")
        title = QLabel("napari preferences")
        title.setObjectName("title")

        self.layout().addWidget(title, 0, 0, 1, 2)
        for i, settup in enumerate(OPTIONS.values()):
            val = bool(SETTINGS.value(settup.key))
            stuff = val_to_widget(val)
            if not stuff:
                continue
            widg, signal, getter, dtype = stuff
            signal.connect(self.set_param(settup.key, getter, type(val)))
            label = ClickableLabel(settup.description)
            label.clicked.connect(widg.toggle)
            self.layout().addWidget(widg, i + 1, 0)
            self.layout().addWidget(label, i + 1, 1)
            self.layout().setSpacing(16)
            self.layout().setContentsMargins(50, 25, 70, 40)
        self.layout().setColumnStretch(1, 1)
        self.setFixedSize(self.sizeHint())

    def set_param(self, key, getter, dtype):
        """ update the parameter dict when the widg has changed """

        def func():
            SETTINGS.setValue(key, dtype(getter()))

        return func

    @property
    def stylesheet(self):
        """doing this here instead of stylesheets.qss because it inherits too
        many bad styles from other objects (until we clean it up)
        """

        return """
        #preferencesWindow {
            background: {{ background }};
        }

        #preferencesWindow > QLabel {
            color: #CCC;
        }

        #preferencesWindow > #title {
            font-size: 18px;
        }
        """
