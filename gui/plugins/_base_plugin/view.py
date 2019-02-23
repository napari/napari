from PyQt5.QtWidgets import (QSlider, QLineEdit, QGridLayout, QFrame,
                             QVBoxLayout, QCheckBox, QWidget, QApplication,
                             QLabel, QComboBox, QPushButton)
from PyQt5.QtCore import Qt

class QtPlugin(QFrame):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.setFixedWidth(200)
        self.grid_layout.setColumnMinimumWidth(0, 100)
        self.grid_layout.setColumnMinimumWidth(1, 100)

        btn = QPushButton()
        btn.setFixedWidth(28)
        btn.setFixedHeight(28)
        btn.setToolTip('Run plugin')
        btn.clicked.connect(self.plugin.run)
        self.grid_layout.addWidget(QLabel('run:'), 0, 0)
        self.grid_layout.addWidget(btn, 0, 1)
