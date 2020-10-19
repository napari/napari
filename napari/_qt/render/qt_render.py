"""QtAsync widget.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget


class QtRender(QWidget):
    """Dockable widget for render controls.

    Attributes
    ----------
    """

    def __init__(self, layer):
        """Create our windgets.
        """
        super().__init__()
        self.layer = layer
        layout = QVBoxLayout()

        spin_layout = QHBoxLayout()

        self.spin_level = QSpinBox()
        self.spin_level.setKeyboardTracking(False)
        self.spin_level.setSingleStep(1)
        self.spin_level.setMinimum(0)
        self.spin_level.setMaximum(10)
        self.spin_level.valueChanged.connect(self.changeSelection)
        self.spin_level.setAlignment(Qt.AlignCenter)

        label = QLabel("Quadtree Level:")
        spin_layout.addWidget(label)
        spin_layout.addWidget(self.spin_level)

        layout.addLayout(spin_layout)
        self.setLayout(layout)

    def changeSelection(self, value):
        """Change currently selected label.

        Parameters
        ----------
        value : int
            Index of label to select.
        """
        self.layer.selected_label = value
        self.selectionSpinBox.clearFocus()
        self.setFocus()
