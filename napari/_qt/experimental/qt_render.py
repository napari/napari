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

        self.layer.events.octree_level.connect(self._on_octree_level)
        layout = QVBoxLayout()

        spin_layout = QHBoxLayout()

        self.spin_level = QSpinBox()
        self.spin_level.setKeyboardTracking(False)
        self.spin_level.setSingleStep(1)
        self.spin_level.setMinimum(0)
        self.spin_level.setMaximum(10)
        self.spin_level.valueChanged.connect(self._on_spin)
        self.spin_level.setAlignment(Qt.AlignCenter)

        label = QLabel("Quadtree Level:")
        spin_layout.addWidget(label)
        spin_layout.addWidget(self.spin_level)

        layout.addLayout(spin_layout)
        self.setLayout(layout)

        # Get initial value.
        self._on_octree_level()

    def _on_spin(self, value):
        """Level spinbox changed.

        Parameters
        ----------
        value : int
            New value of the spinbox
        """
        self.layer.octree_level = value

    def _on_octree_level(self, event=None):
        value = self.layer.octree_level
        self.spin_level.setValue(value)
