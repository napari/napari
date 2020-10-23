"""RenderSpinBox class.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSpinBox


class LabeledSpinBox:
    """A label plus a SpinBox for the QtRender widget.

    This was cobbled together quickly for QtRender. Could we use some better
    napari-standard control here instead?
    """

    def __init__(
        self,
        parent,
        label_text: str,
        initial_value: int,
        spin_range: range,
        connect=None,
    ):
        label = QLabel(label_text)
        self.box = self._create_spin_box(initial_value, spin_range)
        self.connect = connect

        if connect is not None:
            self.box.valueChanged.connect(connect)

        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.box)
        parent.addLayout(layout)

    def _create_spin_box(
        self, initial_value: int, spin_range: range
    ) -> QSpinBox:
        """Return one configured QSpinBox.

        Parameters
        ----------
        initial_value : int
            The initial value of the QSpinBox.
        spin_range : range
            The start/stop/step of the QSpinBox.

        Return
        ------
        QSpinBox
            The configured QSpinBox.
        """
        box = QSpinBox()
        box.setKeyboardTracking(False)
        box.setMinimum(spin_range.start)
        box.setMaximum(spin_range.stop)
        box.setSingleStep(spin_range.step)
        box.setAlignment(Qt.AlignCenter)
        box.setValue(initial_value)
        box.valueChanged.connect(self._on_change)
        return box

    def _on_change(self, value) -> None:
        """Called when the spin box value was changed."""
        # We must clearFocus or it would double-step, no idea why.
        self.box.clearFocus()

        # Notify any connection we have.
        if self.connect is not None:
            self.connect(value)

    def value(self) -> int:
        """Return the current value of the QSpinBox."""
        return self.box.value()

    def set(self, value) -> None:
        """Set the current value of the QSpinBox."""
        self.box.setValue(value)
