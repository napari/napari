from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

MIN_WIDTH = 250
MIN_HEIGHT = 140


class ActivityDialog(QDialog):
    """Activity Dialog for Napari progress bars."""

    resized = Signal(QSize)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Activity')
        self.setMinimumWidth(MIN_WIDTH)
        self.setMinimumHeight(MIN_HEIGHT)
        self.setMaximumHeight(MIN_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.8)
        self.setGraphicsEffect(opacity_effect)

        self.baseWidget = QWidget()

        self.activity_layout = QVBoxLayout()
        self.activity_layout.addStretch()
        self.baseWidget.setLayout(self.activity_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.baseWidget)

        self.title_bar = QLabel()

        title = QLabel('activity', self)
        title.setObjectName('QtCustomTitleLabel')
        title.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        )
        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        title_layout = QHBoxLayout()
        title_layout.setSpacing(4)
        title_layout.setContentsMargins(8, 1, 8, 0)
        line.setFixedHeight(1)
        title_layout.addWidget(line)
        title_layout.addWidget(title)
        self.title_bar.setLayout(title_layout)

        self.base_layout = QVBoxLayout()
        self.base_layout.addWidget(self.title_bar)
        self.base_layout.addWidget(self.scroll)
        self.setLayout(self.base_layout)
