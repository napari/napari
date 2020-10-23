"""QtImageInfo class.
"""
from qtpy.QtWidgets import QFrame, QLabel, QLayout, QVBoxLayout

from .qt_labeled_spin_box import LabeledSpinBox


class QtImageInfo(QFrame):
    """Frame showing an images octree level and dimensions.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.setLayout(self._create_layout())

        # Get initial value and hook to event.
        self._on_octree_level()
        self.layer.events.octree_level.connect(self._on_octree_level)

    def _create_layout(self) -> QLayout:
        """Create layout for image info."""
        layout = QVBoxLayout()
        max_level = self.layer.num_octree_levels - 1
        self.spin_level = LabeledSpinBox(
            layout,
            "Octree Level",
            max_level,
            range(0, max_level, 1),
            connect=self._on_new_level,
        )
        self._add_dimensions(layout)
        return layout

    def _add_dimensions(self, layout: QLayout) -> None:
        """Add dimension labels to layout.

        Parameters
        ----------
        layout : QLayout
            Add dimension labels to this layout.
        """
        shape = self.layer.data.shape
        height, width = shape[1:3]  # which dims are width/height?
        layout.addWidget(QLabel(f"Shape: {shape}"))
        layout.addWidget(QLabel(f"Width: {width}"))
        layout.addWidget(QLabel(f"Height: {height}"))

    def _on_new_level(self, value):
        """Level spinbox changed.

        Parameters
        ----------
        value : int
            New value of the spinbox
        """
        self.layer.octree_level = value

    def _on_octree_level(self, event=None):
        """Set SpinBox to match the layer's new octree_level."""
        value = self.layer.octree_level
        self.spin_level.set(value)
