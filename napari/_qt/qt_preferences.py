from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDialog, QGridLayout, QLabel, QCheckBox

from .auto_widget import val_to_widget
from ..settings import SETTINGS


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
        for i, (key, info) in enumerate(SETTINGS._registered.items()):
            val = SETTINGS.value(key)
            stuff = val_to_widget(val, dtype=info['type'])
            if not stuff:
                continue
            widg, signal, getter, setter = stuff
            signal.connect(self.set_param(key, getter, info['type']))
            label = ClickableLabel(info['description'])
            if hasattr(widg, 'toggle'):
                label.clicked.connect(widg.toggle)
            if isinstance(widg, QCheckBox):
                self.layout().addWidget(widg, i + 1, 0)
                self.layout().addWidget(label, i + 1, 1)
            else:
                self.layout().addWidget(label, i + 1, 0)
                self.layout().addWidget(widg, i + 1, 1)
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
            color: {{ text }};
        }

        #preferencesWindow > #title {
            font-size: 18px;
        }
        """
